"""
Mamba分支共享训练工具模块
为ViT-Fly项目的Mamba分支B-E提供标准化的训练基础设施

功能:
1. 标准化的损失函数 (MSE + 正则化)
2. 配置管理工具 (YAML/JSON配置解析)
3. 检查点保存和加载
4. 训练进度监控 (TensorBoard集成)
5. 学习率调度器
6. 梯度裁剪和优化器管理
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, List, Union


class MambaTrainingConfig:
    """Mamba训练配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径 (支持 .txt, .json, .yaml)
            **kwargs: 覆盖配置参数
        """
        self.default_config = {
            # 基础配置
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'basedir': '/home/vitfly',
            'logdir': 'training/logs',
            'datadir': 'training/datasets',
            'ws_suffix': '',
            
            # 数据配置
            'dataset': 'data',
            'short': 0,
            'val_split': 0.2,
            'seed': 42,
            
            # 模型配置
            'model_type': 'DroneMamba',
            'branch': 'B',  # B, C, D, E
            'load_checkpoint': False,
            'checkpoint_path': '',
            
            # 训练配置
            'lr': 1e-4,
            'N_eps': 50,
            'lr_warmup_epochs': 5,
            'lr_decay': True,
            'lr_decay_rate': 0.95,
            'lr_decay_steps': 10,
            'save_model_freq': 25,
            'val_freq': 10,
            'batch_size': 1,  # 轨迹级别批处理
            'grad_clip': 1.0,
            
            # 损失函数配置
            'loss_type': 'mse',
            'weight_decay': 1e-5,
            'reg_lambda': 0.01,
            
            # 优化器配置
            'optimizer': 'adamw',
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            
            # TensorBoard配置
            'tensorboard': True,
            'log_freq': 10,
            
            # 实验配置
            'experiment_name': '',
            'tags': [],
        }
        
        # 加载配置文件
        if config_path:
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
            
        # 覆盖配置参数
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                
        # 设置设备
        if self.config['device'] == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU")
            self.config['device'] = 'cpu'
            
    def load_config(self, config_path: str):
        """加载配置文件"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        if config_path.suffix == '.txt':
            # 解析txt格式配置 (兼容现有格式)
            self.config = self.default_config.copy()
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 类型转换
                        if key in self.config:
                            if isinstance(self.config[key], bool):
                                self.config[key] = value.lower() == 'true'
                            elif isinstance(self.config[key], int):
                                self.config[key] = int(value)
                            elif isinstance(self.config[key], float):
                                self.config[key] = float(value)
                            else:
                                self.config[key] = value
                                
        elif config_path.suffix in ['.json', '.yaml', '.yml']:
            with open(config_path, 'r') as f:
                if config_path.suffix == '.json':
                    loaded_config = json.load(f)
                else:
                    loaded_config = yaml.safe_load(f)
                    
            # 合并配置
            self.config = self.default_config.copy()
            self.config.update(loaded_config)
            
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
    def save_config(self, save_path: str):
        """保存配置到文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            with open(save_path, 'w') as f:
                for key, value in sorted(self.config.items()):
                    f.write(f"{key} = {value}\n")
                    
    def get_workspace_path(self) -> Path:
        """获取工作空间路径"""
        exp_name = self.config['experiment_name'] or datetime.now().strftime('mamba_%Y%m%d_%H%M%S')
        workspace = Path(self.config['basedir']) / self.config['logdir'] / exp_name
        
        # 添加后缀
        if self.config['ws_suffix']:
            workspace = Path(f"{workspace}{self.config['ws_suffix']}")
            
        # 确保唯一性
        counter = 1
        original_workspace = workspace
        while workspace.exists():
            workspace = Path(f"{original_workspace}_{counter}")
            counter += 1
            
        return workspace
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
        
    def __contains__(self, key):
        return key in self.config


class MambaLossFunction:
    """Mamba分支标准损失函数"""
    
    def __init__(self, config: MambaTrainingConfig):
        self.config = config
        self.loss_type = config['loss_type']
        self.reg_lambda = config['reg_lambda']
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            model: 模型实例 (用于正则化)
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量字典
        """
        loss_dict = {}
        
        # 主损失
        if self.loss_type == 'mse':
            main_loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'l1':
            main_loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            main_loss = F.smooth_l1_loss(predictions, targets)
        else:
            raise ValueError(f"未知的损失类型: {self.loss_type}")
            
        loss_dict['main_loss'] = main_loss.item()
        total_loss = main_loss
        
        # L2正则化
        if model is not None and self.reg_lambda > 0:
            l2_reg = torch.tensor(0., device=predictions.device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            l2_reg = self.reg_lambda * l2_reg
            loss_dict['l2_reg'] = l2_reg.item()
            total_loss = total_loss + l2_reg
            
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_validation_metrics(self,
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor) -> Dict[str, float]:
        """计算验证指标"""
        metrics = {}
        
        # MSE
        metrics['mse'] = F.mse_loss(predictions, targets).item()
        
        # MAE
        metrics['mae'] = F.l1_loss(predictions, targets).item()
        
        # RMSE
        metrics['rmse'] = torch.sqrt(torch.tensor(metrics['mse'])).item()
        
        # 相对误差
        with torch.no_grad():
            rel_error = torch.mean(torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8))
            metrics['relative_error'] = rel_error.item()
            
        return metrics


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, workspace: Path, config: MambaTrainingConfig):
        self.workspace = Path(workspace)
        self.checkpoint_dir = self.workspace / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
    def save_checkpoint(self, 
                       epoch: int,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False):
        """保存检查点"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config.config,
            'metrics': metrics or {},
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            
        return checkpoint_path
        
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """加载检查点"""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新检查点"""
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        return latest_path if latest_path.exists() else None
        
    def get_best_checkpoint(self) -> Optional[Path]:
        """获取最佳检查点"""
        best_path = self.checkpoint_dir / 'checkpoint_best.pt'
        return best_path if best_path.exists() else None


class TrainingMonitor:
    """训练监控器 (TensorBoard集成)"""
    
    def __init__(self, workspace: Path, config: MambaTrainingConfig):
        self.workspace = Path(workspace)
        self.config = config
        self.writer = None
        
        if config['tensorboard']:
            log_dir = self.workspace / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
        }
        
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """记录直方图"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
            
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """记录模型图"""
        if self.writer:
            self.writer.add_graph(model, input_tensor)
            
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """记录学习率"""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_scalar(f'learning_rate/group_{i}', lr, step)
            self.history['learning_rates'].append(lr)
            
    def log_gradients(self, model: nn.Module, step: int):
        """记录梯度"""
        if self.writer:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                    
    def log_training_step(self, 
                         step: int,
                         loss: float,
                         metrics: Dict[str, float],
                         optimizer: torch.optim.Optimizer):
        """记录训练步骤"""
        self.log_scalar('train/loss', loss, step)
        self.log_learning_rate(optimizer, step)
        
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f'train/{metric_name}', metric_value, step)
            
        self.history['train_loss'].append(loss)
        self.history['train_metrics'].append(metrics)
        
    def log_validation_step(self,
                           step: int,
                           loss: float,
                           metrics: Dict[str, float]):
        """记录验证步骤"""
        self.log_scalar('val/loss', loss, step)
        
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f'val/{metric_name}', metric_value, step)
            
        self.history['val_loss'].append(loss)
        self.history['val_metrics'].append(metrics)
        
    def flush(self):
        """刷新写入器"""
        if self.writer:
            self.writer.flush()
            
    def close(self):
        """关闭写入器"""
        if self.writer:
            self.writer.close()
            
    def save_history(self):
        """保存训练历史"""
        history_path = self.workspace / 'training_history.json'
        with open(history_path, 'w') as f:
            # 转换Tensor为Python标量
            serializable_history = {}
            for key, value in self.history.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    serializable_history[key] = value
                else:
                    serializable_history[key] = [float(v) if torch.is_tensor(v) else v for v in value]
            json.dump(serializable_history, f, indent=2)


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, config: MambaTrainingConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_lr(self, step: int) -> float:
        """获取当前学习率"""
        self.current_step = step
        
        # 预热阶段
        if step < self.config['lr_warmup_epochs']:
            warmup_factor = step / self.config['lr_warmup_epochs']
            return self.config['lr'] * warmup_factor
            
        # 衰减阶段
        if self.config['lr_decay']:
            decay_steps = self.config['lr_decay_steps']
            decay_rate = self.config['lr_decay_rate']
            
            decay_epochs = (step - self.config['lr_warmup_epochs']) // decay_steps
            return self.config['lr'] * (decay_rate ** decay_epochs)
            
        return self.config['lr']
        
    def step(self, optimizer: torch.optim.Optimizer):
        """更新学习率"""
        lr = self.get_lr(self.current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        
        return lr


class GradientManager:
    """梯度管理器"""
    
    def __init__(self, config: MambaTrainingConfig):
        self.config = config
        self.grad_norms = []
        
    def clip_gradients(self, model: nn.Module, max_norm: Optional[float] = None):
        """裁剪梯度"""
        if max_norm is None:
            max_norm = self.config['grad_clip']
            
        if max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=max_norm,
                norm_type=2
            )
            self.grad_norms.append(total_norm.item())
            return total_norm
            
        return torch.tensor(0.0)
        
    def get_gradient_stats(self) -> Dict[str, float]:
        """获取梯度统计信息"""
        if not self.grad_norms:
            return {}
            
        return {
            'grad_norm_mean': np.mean(self.grad_norms),
            'grad_norm_std': np.std(self.grad_norms),
            'grad_norm_max': np.max(self.grad_norms),
            'grad_norm_min': np.min(self.grad_norms),
        }


def create_optimizer(model: nn.Module, config: MambaTrainingConfig) -> torch.optim.Optimizer:
    """创建优化器"""
    
    # 获取需要权重衰减的参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 根据参数名判断是否需要权重衰减
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    # 参数分组
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    # 创建优化器
    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config['lr'],
            betas=config['betas'],
            eps=config['eps'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=config['lr'],
            betas=config['betas'],
            eps=config['eps'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_groups,
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"未知的优化器: {config['optimizer']}")
        
    return optimizer


def setup_training_environment(config: MambaTrainingConfig) -> Dict[str, Any]:
    """
    设置训练环境
    
    Returns:
        包含所有训练组件的字典
    """
    # 设置随机种子
    if config['seed'] > 0:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
            
    # 创建工作空间
    workspace = config.get_workspace_path()
    workspace.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config.save_config(workspace / 'config.yaml')
    
    # 初始化组件
    components = {
        'config': config,
        'workspace': workspace,
        'loss_fn': MambaLossFunction(config),
        'checkpoint_manager': CheckpointManager(workspace, config),
        'monitor': TrainingMonitor(workspace, config),
        'gradient_manager': GradientManager(config),
    }
    
    return components