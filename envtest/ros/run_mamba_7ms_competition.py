#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DroneMamba 7m/s 高速避障仿真评估节点

用法:
    python3 run_mamba_7ms_competition.py \
        --model_path path/to/model.pth \
        --num_episodes 50 \
        --output results/mamba_7ms_eval/
"""

import argparse
import rospy
from dodgeros_msgs.msg import Command, QuadState
from envsim_msgs.msg import ObstacleArray
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped

import sys
import os
from os.path import join as opj
import time
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime

# 添加模型路径
sys.path.insert(0, opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import DroneMamba


class DroneMamba7msEvaluator:
    """DroneMamba 7m/s 高速避障评估器"""

    def __init__(self, model_path, output_dir='results/mamba_7ms_eval', desired_velocity=7.0):
        print("="*70)
        print("DroneMamba 7m/s 高速避障评估")
        print("="*70)

        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 评估参数
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.desired_velocity = desired_velocity  # 7m/s
        print(f"\n期望速度：{self.desired_velocity} m/s")

        # 状态变量
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_state = None
        self.is_collided = False
        self.has_finished = False
        self.start_time = None
        self.start_pos = None
        self.prev_pos = None
        self.total_distance = 0.0

        # 数据记录
        self.data_log = {
            'timestamp': [], 'pos_x': [], 'pos_y': [], 'pos_z': [],
            'vel_x': [], 'vel_y': [], 'vel_z': [],
            'velcmd_x': [], 'velcmd_y': [], 'velcmd_z': [],
            'is_collide': [], 'distance_to_goal': [], 'distance_traveled': []
        }

        # 初始化 ROS
        rospy.init_node("mamba_7ms_evaluator", anonymous=False)
        self._init_subscribers()
        self._init_publishers()

        print(f"\n设备：{self.device}")
        print(f"模型参数量：{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"初始化完成，等待仿真启动...\n")

    def load_model(self, model_path):
        """加载模型"""
        print(f"加载模型：{model_path}")
        model = DroneMamba(use_temporal_ssm=False, hidden_size=128)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ 模型加载成功")
        return model

    def _init_subscribers(self):
        self.image_sub = rospy.Subscriber("/kingfisher/depth_image", Image, 
                                           self.callback_image, queue_size=1, tcp_nodelay=True)
        self.state_sub = rospy.Subscriber("/kingfisher/state", QuadState, 
                                           self.callback_state, queue_size=1, tcp_nodelay=True)

    def _init_publishers(self):
        self.command_pub = rospy.Publisher("/kingfisher/command", Command, queue_size=1, tcp_nodelay=True)

    def callback_image(self, msg):
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception as e:
            rospy.logwarn(f"图像处理失败：{e}")

    def callback_state(self, msg):
        self.current_state = msg
        if msg.collision and not self.is_collided:
            self.is_collided = True
            rospy.loginfo("⚠️ 检测到碰撞！")

    def preprocess_input(self):
        if self.current_image is None or self.current_state is None:
            return None

        # 处理深度图像
        depth_img = self.current_image.astype(np.float32) / 255.0
        depth_img = cv2.resize(depth_img, (90, 60))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).to(self.device)

        # 期望速度 7m/s
        desired_vel = torch.tensor([[self.desired_velocity]]).to(self.device)

        # 四元数
        quat = [self.current_state.pose.orientation.w, self.current_state.pose.orientation.x,
                self.current_state.pose.orientation.y, self.current_state.pose.orientation.z]
        quat_tensor = torch.tensor([quat]).to(self.device)

        return [depth_tensor, desired_vel, quat_tensor]

    def compute_command(self):
        """计算速度命令"""
        inputs = self.preprocess_input()
        if inputs is None:
            return None

        with torch.no_grad():
            output, _ = self.model(inputs)

        # 解析输出
        vel_cmd = output.squeeze().cpu().numpy()
        vel_cmd = vel_cmd * self.desired_velocity  # 缩放到 7m/s

        # 创建 Command 消息
        cmd = Command()
        cmd.header.stamp = rospy.Time.now()
        cmd.mode = Command.MODE_VELOCITY
        cmd.velocity = vel_cmd.tolist()
        return cmd

    def run_episode(self, episode_id, timeout=30):
        """运行单次评估"""
        print(f"\n{'='*50}")
        print(f"Episode {episode_id+1:03d}")
        print(f"{'='*50}")

        # 重置状态
        self.is_collided = False
        self.has_finished = False
        self.start_time = None
        self.start_pos = None
        self.prev_pos = None
        self.total_distance = 0.0
        self.data_log = {k: [] for k in self.data_log}

        episode_start = time.time()

        while not rospy.is_shutdown():
            # 检查超时
            if time.time() - episode_start > timeout:
                print(f"⚠️ 超时 ({timeout}s)")
                break

            # 检查碰撞
            if self.is_collided:
                print("❌ 碰撞")
                break

            # 计算并发布速度命令
            cmd = self.compute_command()
            if cmd is not None:
                self.command_pub.publish(cmd)

                # 记录数据
                if self.current_state is not None:
                    self._log_data()

            # 检查是否到达目标
            if self.current_state is not None:
                pos = self.current_state.pose.position
                if pos.x >= 60.0:  # 到达 60m 目标
                    print("✓ 到达目标")
                    break

            time.sleep(0.02)  # 50Hz

        # 保存数据
        self._save_episode_data(episode_id)

        # 返回结果
        success = not self.is_collided and self.current_state is not None and self.current_state.pose.position.x >= 60.0
        return success, self.is_collided, self.data_log

    def _log_data(self):
        pos = self.current_state.pose.position
        vel = self.current_state.velocity.linear

        # 计算距离
        if self.start_pos is None:
            self.start_pos = pos
        if self.prev_pos is None:
            self.prev_pos = pos

        dist_traveled = np.sqrt((pos.x - self.prev_pos.x)**2 + 
                                **(pos.y - self.prev_pos.y)2 + 
                                (pos.z - self.prev_pos.z)**2)
        self.total_distance += dist_traveled
        self.prev_pos = pos

        dist_to_goal = 60.0 - pos.x

        # 速度命令
        vel_cmd = [0, 0, 0]  # 从最后发布的命令获取

        self.data_log['timestamp'].append(time.time() - (self.start_time or time.time()))
        self.data_log['pos_x'].append(pos.x)
        self.data_log['pos_y'].append(pos.y)
        self.data_log['pos_z'].append(pos.z)
        self.data_log['vel_x'].append(vel.x)
        self.data_log['vel_y'].append(vel.y)
        self.data_log['vel_z'].append(vel.z)
        self.data_log['velcmd_x'].append(vel_cmd[0])
        self.data_log['velcmd_y'].append(vel_cmd[1])
        self.data_log['velcmd_z'].append(vel_cmd[2])
        self.data_log['is_collide'].append(1 if self.is_collided else 0)
        self.data_log['distance_to_goal'].append(dist_to_goal)
        self.data_log['distance_traveled'].append(self.total_distance)

    def _save_episode_data(self, episode_id):
        df = pd.DataFrame(self.data_log)
        df.to_csv(opj(self.output_dir, f'episode_{episode_id+1:03d}.csv'), index=False)

    def run_evaluation(self, num_episodes=50, timeout=30):
        """运行完整评估"""
        print(f"\n开始评估，共 {num_episodes} 次飞行\n")

        successes = 0
        collisions = 0

        for i in range(num_episodes):
            success, collision, _ = self.run_episode(i, timeout)
            if success:
                successes += 1
            if collision:
                collisions += 1

            # 打印统计
            if (i + 1) % 10 == 0:
                print(f"\n已运行 {i+1}/{num_episodes} 次")
                print(f"  成功：{successes} ({100*successes/(i+1):.1f}%)")
                print(f"  碰撞：{collisions} ({100*collisions/(i+1):.1f}%)")

            # 重置仿真（需要外部触发）
            rospy.sleep(1.0)

        # 保存统计
        self._save_statistics(num_episodes, successes, collisions)

    def _save_statistics(self, total, successes, collisions):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'desired_velocity': self.desired_velocity,
            'total_episodes': total,
            'successful_flights': successes,
            'collisions': collisions,
            'success_rate': successes / total if total > 0 else 0,
            'collision_rate': collisions / total if total > 0 else 0
        }

        with open(opj(self.output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        # 打印最终结果
        print("\n" + "="*70)
        print("7m/s 高速避障评估完成")
        print("="*70)
        print(f"总飞行次数：{total}")
        print(f"成功：{successes} ({100*successes/total:.1f}%)")
        print(f"碰撞：{collisions} ({100*collisions/total:.1f}%)")
        print(f"结果保存至：{self.output_dir}")
        print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DroneMamba 7m/s 高速避障评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--num_episodes', type=int, default=50, help='评估次数')
    parser.add_argument('--timeout', type=int, default=30, help='超时时间（秒）')
    parser.add_argument('--output_dir', type=str, default='results/mamba_7ms_eval', help='输出目录')
    parser.add_argument('--velocity', type=float, default=7.0, help='期望速度（m/s）')

    args = parser.parse_args()

    evaluator = DroneMamba7msEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        desired_velocity=args.velocity
    )

    try:
        evaluator.run_evaluation(num_episodes=args.num_episodes, timeout=args.timeout)
    except rospy.ROSInterruptException:
        pass
