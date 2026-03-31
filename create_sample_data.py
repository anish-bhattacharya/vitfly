#!/usr/bin/env python3
"""
创建示例数据集用于测试
"""

import os
import numpy as np
import cv2
from pathlib import Path

def create_sample_dataset():
    """创建示例数据集"""
    data_dir = Path("training/datasets/data")
    
    # 创建3个示例轨迹
    for traj_num in range(1, 4):
        traj_dir = data_dir / f"{traj_num:04d}"
        traj_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"创建轨迹 {traj_dir.name}")
        
        # 创建CSV文件
        create_sample_csv(traj_dir)
        
        # 创建图像文件
        create_sample_images(traj_dir)
        
    print(f"示例数据集创建完成，位置: {data_dir}")

def create_sample_csv(traj_dir: Path):
    """创建示例CSV文件"""
    csv_file = traj_dir / "data.csv"
    
    # CSV列头 (根据dataloading.py)
    header = "timestamp,ros_time,desired_vel_x,desired_vel_y,desired_vel_z,q_w,q_x,q_y,q_z,curr_vel_x,curr_vel_y,curr_vel_z,curr_pos_x,curr_pos_y,curr_pos_z,ct_br_x,ct_br_y,ct_br_z,ct_br_w,collision"
    
    # 生成数据
    lines = [header]
    num_frames = 50  # 每个轨迹50帧
    
    for i in range(num_frames):
        timestamp = i * 0.1  # 100ms间隔
        ros_time = 1234567890.0 + timestamp
        
        # 生成随机但合理的数据
        desired_vel_x = 1.0 + np.random.normal(0, 0.1)
        desired_vel_y = np.random.normal(0, 0.05)
        desired_vel_z = 0.0
        
        # 四元数 (轻微随机旋转)
        q_w = 0.99 + np.random.normal(0, 0.01)
        q_x = np.random.normal(0, 0.01)
        q_y = np.random.normal(0, 0.01)
        q_z = np.random.normal(0, 0.01)
        
        # 归一化四元数
        norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
        q_w /= norm
        q_x /= norm
        q_y /= norm
        q_z /= norm
        
        # 当前速度
        curr_vel_x = desired_vel_x + np.random.normal(0, 0.05)
        curr_vel_y = desired_vel_y + np.random.normal(0, 0.02)
        curr_vel_z = 0.0
        
        # 当前位置
        curr_pos_x = i * 0.1 + np.random.normal(0, 0.01)
        curr_pos_y = np.random.normal(0, 0.02)
        curr_pos_z = 1.0 + np.random.normal(0, 0.01)
        
        # 接触点边界框
        ct_br_x = np.random.normal(0, 0.1)
        ct_br_y = np.random.normal(0, 0.1)
        ct_br_z = np.random.normal(0, 0.1)
        ct_br_w = np.random.normal(0, 0.1)
        
        # 碰撞标志 (大多数为0)
        collision = 0
        
        # 构建行
        row = f"{timestamp:.6f},{ros_time:.6f},{desired_vel_x:.6f},{desired_vel_y:.6f},{desired_vel_z:.6f},{q_w:.6f},{q_x:.6f},{q_y:.6f},{q_z:.6f},{curr_vel_x:.6f},{curr_vel_y:.6f},{curr_vel_z:.6f},{curr_pos_x:.6f},{curr_pos_y:.6f},{curr_pos_z:.6f},{ct_br_x:.6f},{ct_br_y:.6f},{ct_br_z:.6f},{ct_br_w:.6f},{collision}"
        lines.append(row)
    
    # 写入文件
    with open(csv_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  创建CSV文件: {csv_file} ({num_frames}行)")

def create_sample_images(traj_dir: Path):
    """创建示例图像文件"""
    num_frames = 50
    
    for i in range(num_frames):
        # 创建60x90的深度图像
        img = np.zeros((60, 90), dtype=np.uint8)
        
        # 添加一些随机障碍物模式
        # 1. 随机点
        for _ in range(20):
            x = np.random.randint(0, 90)
            y = np.random.randint(0, 60)
            img[y, x] = np.random.randint(100, 255)
        
        # 2. 随机线 (模拟墙壁)
        if np.random.random() > 0.5:
            y = np.random.randint(20, 40)
            img[y, :] = np.random.randint(150, 200)
        
        # 3. 随机矩形 (模拟障碍物)
        if np.random.random() > 0.7:
            x1 = np.random.randint(10, 70)
            y1 = np.random.randint(10, 40)
            x2 = x1 + np.random.randint(10, 20)
            y2 = y1 + np.random.randint(10, 20)
            img[y1:y2, x1:x2] = np.random.randint(180, 230)
        
        # 添加一些高斯噪声
        noise = np.random.normal(0, 10, (60, 90)).astype(np.int32)
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        # 保存图像
        timestamp = i * 0.1
        img_file = traj_dir / f"{timestamp:.6f}.png"
        cv2.imwrite(str(img_file), img)
    
    print(f"  创建图像文件: {num_frames}个PNG文件")

if __name__ == "__main__":
    create_sample_dataset()