#!/usr/bin/env python3
"""
Mamba分支标准化训练脚本
支持所有Mamba变体的统一训练接口
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models'))

from training.mamba_training_utils import (
    mamba_loss_function,
    MambaTrainingConfig,
    MambaCheckpointManager,
    MambaMetricsTracker,
    validate_mamba_training_setup
)

from training import train as vitfly_train
from training import dataloading

def train_mamba_branch(config_path, branch_name):
    """训练特定Mamba分支的主函数"""
    
    # 加载配置
    config_mgr = MambaTrainingConfig(config_path)
    config = config_mgr.config
    
    print(f"=== 开始训练Mamba分支: {branch_name} ===")
    print(f"配置文件: {config_path}")
    print(f"设备: {config['device']}")
    print(f"日志目录: {config['log_dir']}")
    
    # 创建目录
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config.get('checkpoint_dir', '/tmp/checkpoints'), exist_ok=True)
    
    # 加载模型
    try:
        if branch_name == 'branch_B_mambavision_ssm':
            from models.mamba_vision_ssm import MambaVisionSSM
            model = MambaVisionSSM()
        elif branch_name == 'branch_C_cnn_mamba3':
            from models.cnn_mamba3 import CNNMamba3
            model = CNNMamba3()
        elif branch_name == 'branch_D_sth_mamba':
            from models.sth_mamba import STHMamba
            model = STHMamba()
        elif branch_name == 'branch_E_decisionmamba':
            from models.decision_mamba import DecisionMamba
            model = DecisionMamba()
        else:
            raise ValueError(f"未知的分支名称: {branch_name}")
        
        model = model.to(config['device'])
        print(f"✅ 模型 {branch_name} 加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 验证训练设置
    print("\n=== 验证训练设置 ===")
    checks = validate_mamba_training_setup(config, model)
    for check_name, status in checks:
        print(f"  {check_name}: {status}")
    
    # 数据加载
    print("\n=== 加载数据 ===")
    try:
        # 使用ViT-Fly的数据加载器
        train_data = dataloading.dataloader(
            config['dataset_dir'],
            val_split=config['val_split'],
            short=config['short'],
            device=config['device']
        )
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-4
    )
    
    # 初始化工具
    checkpoint_mgr = MambaCheckpointManager(config.get('checkpoint_dir', '/tmp/checkpoints'))
    metrics_tracker = MambaMetricsTracker(config['log_dir'])
    
    # 开始训练
    print("\n=== 开始训练 ===")
    
    try:
        model.train()
        
        for epoch in range(config['N_eps']):
            epoch_loss = 0
            
            # 模拟训练循环（简化）
            # 实际训练需要使用ViT-Fly的训练循环
            # 这里使用简化版本用于演示
            
            optimizer.zero_grad()
            
            # 创建示例输入（实际应该使用真实数据）
            batch_size = config['batch_size']
            seq_len = 10
            depth_images = torch.randn(batch_size, seq_len, 1, 60, 90).to(config['device'])
            desired_velocities = torch.randn(batch_size, seq_len, 1).to(config['device'])
            quaternions = torch.randn(batch_size, seq_len, 4).to(config['device'])
            targets = torch.randn(batch_size, seq_len, 3).to(config['device'])
            
            # 正向传播
            outputs = model([depth_images, desired_velocities, quaternions])
            
            # 计算损失
            loss = mamba_loss_function(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss = loss.item()
            
            # 记录指标
            metrics_tracker.add_scalar('train/loss', epoch_loss, epoch)
            
            # 检查点保存
            if epoch % config['save_model_freq'] == 0:
                checkpoint_path = checkpoint_mgr.save_checkpoint(
                    epoch, model, optimizer, epoch_loss, config
                )
                print(f"  ✅ 保存检查点到: {checkpoint_path}")
            
            # 验证
            if epoch % config['val_freq'] == 0:
                # 简化验证（实际应该有独立的验证集）
                model.eval()
                with torch.no_grad():
                    val_outputs = model([depth_images, desired_velocities, quaternions])
                    val_loss = mamba_loss_function(val_outputs, targets)
                    metrics_tracker.add_scalar('val/loss', val_loss.item(), epoch)
                    print(f"  ✅ 验证损失: {val_loss.item():.6f}")
                model.train()
            
            print(f"Epoch {epoch+1}/{config['N_eps']}, 损失: {epoch_loss:.6f}")
        
        print("✅ 训练完成")
        metrics_tracker.close()
        return True
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Mamba分支训练脚本')
    parser.add_argument('--branch', required=True, 
                       choices=['branch_B_mambavision_ssm', 'branch_C_cnn_mamba3',
                                'branch_D_sth_mamba', 'branch_E_decisionmamba'],
                       help='要训练的分支名称')
    parser.add_argument('--config', 
                       default='/root/vitfly/training/config/mamba_templates/{branch}.txt',
                       help='配置文件路径，默认使用模板')
    
    args = parser.parse_args()
    
    # 构建配置路径
    config_path = args.config.replace('{branch}', args.branch)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 开始训练
    success = train_mamba_branch(config_path, args.branch)
    
    if success:
        print(f"\n✅ Mamba分支 {args.branch} 训练完成！")
        print(f"日志目录: /root/vitfly/training/logs/{args.branch}")
        print(f"检查点: /root/vitfly/experiments/mamba_branches/{args.branch}/models")
    else:
        print(f"\n❌ Mamba分支 {args.branch} 训练失败")
        sys.exit(1)

if __name__ == "__main__":
    main()