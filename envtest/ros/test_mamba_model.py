#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DroneMamba ROS 测试节点 - 不依赖 Gazebo 的简化测试
验证模型可以加载并生成合理的速度命令
"""

import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
import sys
import os

# 添加模型路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../models'))
import model as model_library

class MambaTesterNode:
    def __init__(self):
        rospy.init_node('mamba_tester', anonymous=True)
        
        # 加载模型
        model_path = rospy.get_param('~model_path', '/root/.lingma/worktree/vitfly/XBSDYR/checkpoints/drone_mamba_latest.pth')
        self.use_cuda = rospy.get_param('~use_cuda', True) and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        rospy.loginfo(f"Loading model from {model_path} on {self.device}")
        
        self.model = model_library.DroneMamba(use_temporal_ssm=True, d_state=8, hidden_size=128)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        rospy.loginfo(f"Model loaded successfully! Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 统计信息
        self.num_inferences = 0
        self.total_latency = 0.0
        self.velocity_commands = []
        
        # 发布速度命令
        self.cmd_pub = rospy.Publisher('/kingfisher/command', Twist, queue_size=10)
        
        # 订阅深度图像（模拟）
        self.depth_sub = rospy.Subscriber('/kingfisher/depth_image', Image, self.depth_callback, queue_size=10)
        
        # 订阅状态
        self.state_sub = rospy.Subscriber('/kingfisher/state', Odometry, self.state_callback, queue_size=10)
        
        # 当前状态
        self.current_depth = None
        self.current_state = None
        self.desvel = 5.0  # 默认期望速度
        
        rospy.loginfo("Mamba tester node ready. Waiting for data...")
        
    def depth_callback(self, msg):
        """处理深度图像"""
        try:
            # 将 ROS 图像转换为 tensor
            depth_array = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
            self.current_depth = depth_tensor.to(self.device)
        except Exception as e:
            rospy.logwarn(f"Error processing depth image: {e}")
    
    def state_callback(self, msg):
        """处理无人机状态"""
        self.current_state = msg
        
    def run_inference(self):
        """运行一次推理"""
        if self.current_depth is None:
            return None
        
        try:
            # 准备输入
            batch_size = 1
            seq_len = 1
            images = self.current_depth.repeat(seq_len, batch_size, 1, 1, 1)  # [T, B, C, H, W]
            desvel = torch.ones(seq_len, batch_size).to(self.device) * self.desvel
            
            # 四元数
            if self.current_state:
                q = self.current_state.pose.pose.orientation
                currquat = torch.tensor([[[q.w, q.x, q.y, q.z]]]).repeat(seq_len, 1, 1).to(self.device)
            else:
                currquat = torch.zeros(seq_len, batch_size, 4).to(self.device)
                currquat[:, :, 0] = 1.0
            
            inputs = [images, desvel, currquat]
            
            # 推理
            start = rospy.Time.now()
            with torch.no_grad():
                output, _ = self.model(inputs)
            inference_time = (rospy.Time.now() - start).to_sec() * 1000  # ms
            
            # 更新统计
            self.num_inferences += 1
            self.total_latency += inference_time
            
            # 解析输出
            vel_cmd = output[0, 0].cpu().numpy()  # [vx, vy, vz]
            self.velocity_commands.append(vel_cmd)
            
            return vel_cmd, inference_time
            
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")
            return None
    
    def spin(self):
        """主循环"""
        rate = rospy.Rate(20)  # 20 Hz
        
        while not rospy.is_shutdown():
            if self.current_depth is not None:
                result = self.run_inference()
                
                if result:
                    vel_cmd, latency = result
                    
                    # 发布速度命令
                    twist = Twist()
                    twist.linear.x = float(vel_cmd[0])
                    twist.linear.y = float(vel_cmd[1])
                    twist.linear.z = float(vel_cmd[2])
                    self.cmd_pub.publish(twist)
                    
                    if self.num_inferences % 10 == 0:
                        avg_latency = self.total_latency / self.num_inferences
                        rospy.loginfo(f"Inference #{self.num_inferences}: latency={latency:.2f}ms, avg={avg_latency:.2f}ms, cmd=[{vel_cmd[0]:.3f}, {vel_cmd[1]:.3f}, {vel_cmd[2]:.3f}]")
            
            rate.sleep()
    
    def print_statistics(self):
        """打印统计信息"""
        if self.num_inferences > 0:
            avg_latency = self.total_latency / self.num_inferences
            velocity_commands = np.array(self.velocity_commands)
            
            rospy.loginfo("="*70)
            rospy.loginfo("Mamba Tester Statistics")
            rospy.loginfo("="*70)
            rospy.loginfo(f"Total inferences: {self.num_inferences}")
            rospy.loginfo(f"Average latency: {avg_latency:.2f}ms ({1000/avg_latency:.1f} FPS)")
            rospy.loginfo(f"Velocity commands:")
            rospy.loginfo(f"  Mean: [{velocity_commands[:, 0].mean():.3f}, {velocity_commands[:, 1].mean():.3f}, {velocity_commands[:, 2].mean():.3f}]")
            rospy.loginfo(f"  Std:  [{velocity_commands[:, 0].std():.3f}, {velocity_commands[:, 1].std():.3f}, {velocity_commands[:, 2].std():.3f}]")
            rospy.loginfo(f"  Range: [{velocity_commands.min():.3f}, {velocity_commands.max():.3f}]")
            rospy.loginfo("="*70)

if __name__ == '__main__':
    try:
        node = MambaTesterNode()
        node.spin()
        node.print_statistics()
    except rospy.ROSInterruptException:
        pass
