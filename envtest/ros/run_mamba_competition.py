#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DroneMamba 仿真评估节点

用法:
    python3 run_mamba_competition.py \
        --model_path path/to/model.pth \
        --model_type DroneMamba \
        --num_episodes 50 \
        --output results/mamba_eval/
"""

import argparse
import rospy
from dodgeros_msgs.msg import Command, QuadState
from envsim_msgs.msg import ObstacleArray
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
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


class DroneMambaEvaluator:
    """DroneMamba 仿真评估器"""
    
    def __init__(self, model_path, model_type='DroneMamba', output_dir='results/mamba_eval'):
        print("[DroneMambaEvaluator] 初始化...")
        
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 评估参数
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 状态变量
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_state = None
        self.obstacles = None
        self.is_collided = False
        self.has_finished = False
        
        # 数据记录
        self.data_log = {
            'timestamp': [],
            'pos_x': [], 'pos_y': [], 'pos_z': [],
            'vel_x': [], 'vel_y': [], 'vel_z': [],
            'velcmd_x': [], 'velcmd_y': [], 'velcmd_z': [],
            'is_collide': [],
            'distance_to_goal': [],
            'distance_traveled': []
        }
        
        self.start_time = None
        self.start_pos = None
        self.total_distance = 0.0
        self.prev_pos = None
        
        # 初始化 ROS
        rospy.init_node("mamba_evaluator", anonymous=False)
        
        # 订阅话题
        self._init_subscribers()
        self._init_publishers()
        
        print(f"[DroneMambaEvaluator] 初始化完成 (设备：{self.device})")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型：{model_path}")
        
        # 始终使用 Temporal SSM 版本（DroneMamba 默认配置）
        model = DroneMamba(use_temporal_ssm=True, d_state=8, hidden_size=128)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ 模型加载成功")
        return model
    
    def _init_subscribers(self):
        """初始化订阅者"""
        self.image_sub = rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/unity/depth",
            Image,
            self.callback_image,
            queue_size=1,
            tcp_nodelay=True
        )
        
        self.state_sub = rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/groundtruth/state",
            QuadState,
            self.callback_state,
            queue_size=1,
            tcp_nodelay=True
        )
        
        self.obstacle_sub = rospy.Subscriber(
            "/kingfisher/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.callback_obstacles,
            queue_size=1,
            tcp_nodelay=True
        )
    
    def _init_publishers(self):
        """初始化发布者"""
        self.command_pub = rospy.Publisher(
            "/kingfisher/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
            tcp_nodelay=True
        )
    
    def callback_image(self, msg):
        """深度图像回调"""
        try:
            # 深度图像是 32FC1 格式
            if msg.encoding == '32FC1':
                depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                # 归一化到 0-1（假设最大深度 10m）
                depth_img = np.clip(depth_array / 10.0, 0, 1)
                self.current_image = (depth_img * 255).astype(np.uint8)
            else:
                self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception as e:
            rospy.logwarn_throttle(5, f"图像处理失败：{e}")
    
    def callback_state(self, msg):
        """状态回调"""
        self.current_state = msg
        
        # 碰撞检测 - 通过速度或高度异常判断
        # QuadState 没有直接的 collision 标志
        if not self.is_collided:
            # 检查是否接近地面（z < 0.5m 可能表示碰撞）
            if msg.pose.position.z < 0.3:
                self.is_collided = True
                rospy.loginfo("⚠️ 检测到碰撞（高度过低）！")
    
    def callback_obstacles(self, msg):
        """障碍物回调"""
        self.obstacles = msg
    
    def preprocess_input(self):
        """预处理输入"""
        if self.current_image is None or self.current_state is None:
            return None
        
        # 处理深度图像
        depth_img = self.current_image.astype(np.float32) / 255.0
        depth_img = cv2.resize(depth_img, (90, 60))  # Resize to 60x90
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 90)
        
        # 期望速度（从状态或固定值）
        desired_vel = torch.tensor([5.0]).view(1, 1)  # 默认 5.0 m/s（高速）
        
        # 四元数 - QuadState 的 pose 是 PoseStamped
        quat = [
            self.current_state.pose.pose.orientation.w,
            self.current_state.pose.pose.orientation.x,
            self.current_state.pose.pose.orientation.y,
            self.current_state.pose.pose.orientation.z
        ]
        quat_tensor = torch.tensor(quat).unsqueeze(0)  # (1, 4)
        
        # 移动到设备
        depth_tensor = depth_tensor.to(self.device)
        desired_vel = desired_vel.to(self.device)
        quat_tensor = quat_tensor.to(self.device)
        
        return [depth_tensor, desired_vel, quat_tensor]
    
    def predict_command(self):
        """预测速度命令"""
        inputs = self.preprocess_input()
        
        if inputs is None:
            return None
        
        with torch.no_grad():
            output, _ = self.model(inputs)
        
        # 转换为速度命令
        vel_cmd = output.cpu().numpy()[0]  # (3,)
        
        return vel_cmd
    
    def log_data(self):
        """记录当前数据"""
        if self.current_state is None:
            return
        
        timestamp = rospy.get_rostime().to_sec()
        
        # 位置
        pos_x = self.current_state.pose.position.x
        pos_y = self.current_state.pose.position.y
        pos_z = self.current_state.pose.position.z
        
        # 速度
        vel_x = self.current_state.twist.linear.x
        vel_y = self.current_state.twist.linear.y
        vel_z = self.current_state.twist.linear.z
        
        # 预测的速度命令
        vel_cmd = self.predict_command()
        if vel_cmd is not None:
            velcmd_x, velcmd_y, velcmd_z = vel_cmd
        else:
            velcmd_x, velcmd_y, velcmd_z = 0.0, 0.0, 0.0
        
        # 距离计算
        if self.start_pos is None:
            self.start_pos = np.array([pos_x, pos_y, pos_z])
        
        current_pos = np.array([pos_x, pos_y, pos_z])
        distance_to_goal = np.linalg.norm(current_pos - self.start_pos - np.array([50, 0, 0]))
        
        if self.prev_pos is not None:
            self.total_distance += np.linalg.norm(current_pos - self.prev_pos)
        self.prev_pos = current_pos
        
        # 记录
        self.data_log['timestamp'].append(timestamp)
        self.data_log['pos_x'].append(pos_x)
        self.data_log['pos_y'].append(pos_y)
        self.data_log['pos_z'].append(pos_z)
        self.data_log['vel_x'].append(vel_x)
        self.data_log['vel_y'].append(vel_y)
        self.data_log['vel_z'].append(vel_z)
        self.data_log['velcmd_x'].append(velcmd_x)
        self.data_log['velcmd_y'].append(velcmd_y)
        self.data_log['velcmd_z'].append(velcmd_z)
        self.data_log['is_collide'].append(1 if self.is_collided else 0)
        self.data_log['distance_to_goal'].append(distance_to_goal)
        self.data_log['distance_traveled'].append(self.total_distance)
    
    def run_episode(self, episode_id, timeout=60):
        """运行单次评估"""
        print(f"\n{'='*60}")
        print(f"Episode {episode_id + 1}")
        print(f"{'='*60}")
        
        # 重置状态
        self.is_collided = False
        self.has_finished = False
        self.start_time = None
        self.start_pos = None
        self.total_distance = 0.0
        self.prev_pos = None
        
        # 清空数据日志
        for key in self.data_log.keys():
            self.data_log[key] = []
        
        # 等待初始化
        rospy.sleep(2.0)
        
        start_time = time.time()
        log_counter = 0
        
        rate = rospy.Rate(50)  # 50 Hz
        
        while not rospy.is_shutdown():
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 超时检查
            if elapsed > timeout:
                print(f"⏰ 超时 ({timeout}s)")
                break
            
            # 碰撞检查
            if self.is_collided:
                print(f"💥 碰撞！飞行时间：{elapsed:.2f}s")
                break
            
            # 预测并发布命令
            vel_cmd = self.predict_command()
            
            if vel_cmd is not None:
                cmd_msg = TwistStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.twist.linear.x = vel_cmd[0]
                cmd_msg.twist.linear.y = vel_cmd[1]
                cmd_msg.twist.linear.z = vel_cmd[2]
                
                self.command_pub.publish(cmd_msg)
            
            # 记录数据（每 0.03 秒）
            if current_time - log_counter >= 0.03:
                self.log_data()
                log_counter = current_time
            
            rate.sleep()
        
        # 保存本次 episode 数据
        df = pd.DataFrame(self.data_log)
        episode_file = opj(self.output_dir, f'episode_{episode_id + 1:03d}.csv')
        df.to_csv(episode_file, index=False)
        
        # 计算指标
        success = not self.is_collided and elapsed < timeout
        flight_time = elapsed
        final_distance = self.data_log['distance_to_goal'][-1] if self.data_log['distance_to_goal'] else 0
        
        result = {
            'episode_id': episode_id,
            'success': success,
            'collision': self.is_collided,
            'timeout': elapsed >= timeout,
            'flight_time': flight_time,
            'final_distance': final_distance,
            'total_distance': self.total_distance
        }
        
        print(f"结果：{'✅ 成功' if success else '❌ 失败'} | "
              f"时间：{flight_time:.2f}s | "
              f"距离：{final_distance:.2f}m")
        
        return result
    
    def run_evaluation(self, num_episodes=50):
        """运行完整评估"""
        print(f"\n{'='*70}")
        print(f"DroneMamba 仿真评估")
        print(f"{'='*70}")
        print(f"模型：{self.device}")
        print(f"Episode 数量：{num_episodes}")
        print(f"输出目录：{self.output_dir}")
        print(f"{'='*70}\n")
        
        results = []
        
        for i in range(num_episodes):
            result = self.run_episode(i)
            results.append(result)
            
            # 每 10 个 episode 保存统计
            if (i + 1) % 10 == 0:
                self.save_statistics(results)
        
        # 最终统计
        self.save_statistics(results)
        self.generate_report(results)
        
        return results
    
    def save_statistics(self, results):
        """保存统计数据"""
        total = len(results)
        successes = sum(1 for r in results if r['success'])
        collisions = sum(1 for r in results if r['collision'])
        timeouts = sum(1 for r in results if r['timeout'])
        
        avg_flight_time = np.mean([r['flight_time'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])
        
        stats = {
            'total_episodes': total,
            'successes': successes,
            'collisions': collisions,
            'timeouts': timeouts,
            'success_rate': successes / total * 100,
            'collision_rate': collisions / total * 100,
            'avg_flight_time': avg_flight_time,
            'avg_final_distance': avg_final_distance
        }
        
        # 保存为 JSON
        stats_file = opj(self.output_dir, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 打印统计
        print(f"\n{'='*50}")
        print(f"统计结果 ({total} episodes)")
        print(f"{'='*50}")
        print(f"成功率：     {stats['success_rate']:.1f}% ({successes}/{total})")
        print(f"碰撞率：     {stats['collision_rate']:.1f}% ({collisions}/{total})")
        print(f"超时率：     {timeouts/total*100:.1f}% ({timeouts}/{total})")
        print(f"平均时间：   {avg_flight_time:.2f}s")
        print(f"平均距离：   {avg_final_distance:.2f}m")
        print(f"{'='*50}\n")
    
    def generate_report(self, results):
        """生成评估报告"""
        report_file = opj(self.output_dir, 'evaluation_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DroneMamba 仿真评估报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总 Episode 数：{len(results)}\n\n")
            
            # 详细统计
            successes = sum(1 for r in results if r['success'])
            collisions = sum(1 for r in results if r['collision'])
            
            f.write("性能指标:\n")
            f.write(f"  - 成功率：{successes/len(results)*100:.1f}%\n")
            f.write(f"  - 碰撞率：{collisions/len(results)*100:.1f}%\n")
            f.write(f"  - 平均飞行时间：{np.mean([r['flight_time'] for r in results]):.2f}s\n")
            f.write(f"  - 平均飞行距离：{np.mean([r['total_distance'] for r in results]):.2f}m\n\n")
            
            # 逐 episode 结果
            f.write("\nEpisode 详细结果:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'ID':>5} | {'结果':>8} | {'时间 (s)':>10} | {'距离 (m)':>12} | {'原因':>15}\n")
            f.write("-"*70 + "\n")
            
            for r in results:
                result_str = "成功" if r['success'] else "失败"
                reason = "完成" if r['success'] else ("碰撞" if r['collision'] else "超时")
                f.write(f"{r['episode_id']:>5} | {result_str:>8} | {r['flight_time']:>10.2f} | {r['total_distance']:>12.2f} | {reason:>15}\n")
            
            f.write("-"*70 + "\n")
        
        print(f"✓ 评估报告已保存：{report_file}")


def main():
    parser = argparse.ArgumentParser(description='DroneMamba 仿真评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--model_type', type=str, default='DroneMamba', 
                        choices=['DroneMamba', 'DroneMamba_SSM'], help='模型类型')
    parser.add_argument('--num_episodes', type=int, default=50, help='评估次数')
    parser.add_argument('--output_dir', type=str, default='results/mamba_eval', help='输出目录')
    parser.add_argument('--timeout', type=int, default=60, help='超时时间 (秒)')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = DroneMambaEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    
    # 运行评估
    try:
        results = evaluator.run_evaluation(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        print("\n⚠️ 评估中断")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("✅ 仿真评估完成！")
    print("="*70)
    print(f"结果保存在：{args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
