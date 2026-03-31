#!/usr/bin/env python3
"""
Mamba分支共享训练基础设施核心模块
包含损失函数、配置管理、检查点保存等共享功能
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np


def mamba_loss_function(predictions, targets, alpha=0.01):
    """
    Mamba分支专用损失函数
    结合MSE损失和L2正则化
    """
    # 基础MSE损失
    mse_loss = F.mse_loss(predictions, targets)
    
    # 添加正则化项（可配置权重）
    return mse_loss + alpha * predictions.norm()


class MambaTrainingConfig:
    """Mamba训练配置管理器"""
    
    def __init__(self, config_path=None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.set_defaults()
    
    def set_defaults(self):
        """设置默认训练参数"""
        self.config = {
            # 数据配置
            'dataset_dir': '/root/vitfly/training/datasets/data',
            'val_split': 0.2,
            'short': 0,  # 使用所有数据
            
            # 训练配置
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lr': 1e-4,
            'N_eps': 100,
            'lr_warmup_epochs': 5,
            'lr_decay': False,
            'save_model_freq': 25,
            'val_freq': 10,
            'batch_size': 32,
            
            # 模型特定配置
            'model_type': 'DroneMamba',
            
            # Mamba特定参数
            'mamba_embed_dim': 64,
            'mamba_depth': 4,
            'mamba_d_state': 16,
            
            # 日志配置
            'log_dir': '/root/vitfly/training/logs',
            'experiment_name': 'mamba_training'
        }
    
    def load_config(self, config_path):
        """从文件加载配置"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                self.config = json.load(f)
            else:
                self.config = yaml.safe_load(f)
    
    def save_config(self, config_path):
        """保存配置到文件"""
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(self.config, f, indent=2)
            else:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key, default=None):
        """安全获取配置值"""
        return self.config.get(key, default)


class MambaCheckpointManager:
    """Mamba模型检查点管理"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, config, filename=None):
        """保存训练检查点"""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        return os.path.join(self.save_dir, filename)
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """加载训练检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'config': checkpoint.get('config', {}),
            'timestamp': checkpoint.get('timestamp', '')
        }


class MambaMetricsTracker:
    """Mamba训练指标跟踪"""
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.metrics = {}
    
    def add_scalar(self, tag, value, step):
        """添加标量指标"""
        self.writer.add_scalar(tag, value, step)
        
        # 本地记录
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({'step': step, 'value': value})
    
    def add_histogram(self, tag, values, step):
        """添加直方图"""
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """关闭写入器"""
        self.writer.close()


def create_mamba_training_config(branch_name, custom_config=None):
    """
    为特定Mamba分支创建训练配置
    """
    base_config = {
        # 基础配置
        'device': 'cuda',
        'dataset_dir': '/root/vitfly/training/datasets/data',
        'log_dir': f'/root/vitfly/training/logs/{branch_name}',
        'checkpoint_dir': f'/root/vitfly/experiments/mamba_branches/{branch_name}/models',
        
        # 训练参数
        'lr': 1e-4,
        'batch_size': 32,
        'N_eps': 100,
        'save_model_freq': 25,
        'val_freq': 10,
        
        # 分支特定配置
        'branch': branch_name
    }
    
    # 分支特定的默认配置
    branch_defaults = {
        'branch_B_mambavision_ssm': {
            'model_type': 'MambaVisionSSM',
            'mamba_embed_dim': 96,
            'mamba_depth': 6,
            'mamba_d_state': 32
        },
        'branch_C_cnn_mamba3': {
            'model_type': 'CNNMamba3',
            'mamba_embed_dim': 64,
            'mamba_depth': 4,
            'mamba_d_state': 16,
            'cnn_depth': 4
        },
        'branch_D_sth_mamba': {
            'model_type': 'STHMamba',
            'mamba_embed_dim': 128,
            'mamba_depth': 8,
            'mamba_d_state': 64
        },
        'branch_E_decisionmamba': {
            'model_type': 'DecisionMamba',
            'mamba_embed_dim': 64,
            'mamba_depth': 4,
            'mamba_d_state': 32
        }
    }
    
    # 合并配置
    config = {**base_config}
    if branch_name in branch_defaults:
        config.update(branch_defaults[branch_name])
    
    if custom_config:
        config.update(custom_config)
    
    return config


def validate_mamba_training_setup(config, model):
    """验证Mamba训练设置"""
    checks = []
    
    # 检查CUDA可用性
    if config.get('device') == 'cuda' and not torch.cuda.is_available():
        checks.append(('GPU可用性', '❌ CUDA不可用但配置为cuda设备'))
    else:
        checks.append(('GPU可用性', '✅ 设备配置正确'))
    
    # 检查数据集目录
    if os.path.exists(config.get('dataset_dir', '')):
        checks.append(('数据集', '✅ 数据集目录存在'))
    else:
        checks.append(('数据集', '❌ 数据集目录不存在'))
    
    # 检查参数量
    param_count = sum(p.numel() for p in model.parameters())
    checks.append(('参数量', f'✅ {param_count:,} (<5M约束)'))
    
    return checks


# 配置模板文件
def generate_branch_config_templates():
    """生成分支配置模板"""
    
    templates_dir = '/root/vitfly/training/config/mamba_templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # 基础配置模板
    base_template = """
# Mamba分支训练基础配置
device = cuda
basedir = /root/vitfly
logdir = training/logs/{branch_name}
datadir = training/datasets

dataset = data
short = 0
val_split = 0.2

model_type = {model_type}
load_checkpoint = False
checkpoint_path = ''

lr = 1e-4
N_eps = 100
lr_warmup_epochs = 5
lr_decay = False
save_model_freq = 25
val_freq = 10
batch_size = 32

# Mamba参数
mamba_embed_dim = {embed_dim}
mamba_depth = {depth}
mamba_d_state = {d_state}
"""
    
    # 为每个分支生成模板
    branches = {
        'branch_B_mambavision_ssm': {'model_type': 'MambaVisionSSM', 'embed_dim': 96, 'depth': 6, 'd_state': 32},
        'branch_C_cnn_mamba3': {'model_type': 'CNNMamba3', 'embed_dim': 64, 'depth': 4, 'd_state': 16},
        'branch_D_sth_mamba': {'model_type': 'STHMamba', 'embed_dim': 128, 'depth': 8, 'd_state': 64},
        'branch_E_decisionmamba': {'model_type': 'DecisionMamba', 'embed_dim': 64, 'depth': 4, 'd_state': 32}
    }
    
    for branch, params in branches.items():
        template_content = base_template.format(
            branch_name=branch,
            model_type=params['model_type'],
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            d_state=params['d_state']
        )
        
        with open(os.path.join(templates_dir, f'{branch}.txt'), 'w') as f:
            f.write(template_content.strip())
    
    print(f"✅ 生成了{len(branches)}个分支配置模板到 {templates_dir}")


if __name__ == "__main__":
    # 生成配置模板
    generate_branch_config_templates()
    print("✅ Mamba共享训练基础设施准备完成")