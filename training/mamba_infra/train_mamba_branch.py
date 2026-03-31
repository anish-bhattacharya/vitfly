#!/usr/bin/env python3
"""
Mamba分支标准化训练脚本
支持所有Mamba分支(B-E)的统一训练接口

功能:
1. 配置驱动的训练参数
2. 自动验证集分割
3. 训练日志和可视化
4. 多分支模型支持
5. 与现有ViT-Fly训练框架集成
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'models'))
sys.path.append(str(project_root / 'training'))

# 导入共享训练工具
from mamba_infra.mamba_training_utils import (
    MambaTrainingConfig,
    setup_training_environment,
    create_optimizer,
    MambaLossFunction,
    LearningRateScheduler,
    GradientManager
)

# 导入现有训练组件
from dataloading import dataloader
import model as model_library


class MambaBranchTrainer:
    """Mamba分支训练器"""
    
    def __init__(self, config_path: str, **kwargs):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
            **kwargs: 覆盖配置参数
        """
        # 加载配置
        self.config = MambaTrainingConfig(config_path, **kwargs)
        
        # 设置训练环境
        self.env = setup_training_environment(self.config)
        self.workspace = self.env['workspace']
        
        # 初始化组件
        self.device = torch.device(self.config['device'])
        self.loss_fn = self.env['loss_fn']
        self.checkpoint_manager = self.env['checkpoint_manager']
        self.monitor = self.env['monitor']
        self.gradient_manager = self.env['gradient_manager']
        
        # 训练状态
        self.current_epoch = 0
        self.total_steps = 0
        self.best_val_loss = float('inf')
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        
        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # 日志文件
        self.log_file = open(self.workspace / 'training.log', 'w')
        
    def setup_data(self):
        """设置数据加载器"""
        print(f"[数据] 加载数据集: {self.config['dataset']}")
        
        # 使用现有的数据加载器
        data_dir = Path(self.config['datadir']) / self.config['dataset']
        
        # 加载所有数据
        train_val_dirs = None
        if self.config['load_checkpoint'] and self.config['checkpoint_path']:
            # 从检查点加载数据分割
            checkpoint = torch.load(self.config['checkpoint_path'], map_location='cpu')
            if 'train_val_dirs' in checkpoint:
                train_val_dirs = checkpoint['train_val_dirs']
        
        # 加载数据
        data = dataloader(
            data_dir=str(data_dir),
            val_split=self.config['val_split'],
            short=self.config['short'],
            seed=self.config['seed'],
            train_val_dirs=train_val_dirs
        )
        
        # 解包数据
        (train_ims, train_desvel, train_currquat, train_velcmd, train_trajlength,
         val_ims, val_desvel, val_currquat, val_velcmd, val_trajlength,
         train_val_dirs) = data
        
        # 转换为Tensor并移动到设备
        self.train_data = {
            'ims': torch.from_numpy(train_ims).float().to(self.device),
            'desvel': torch.from_numpy(train_desvel).float().to(self.device),
            'currquat': torch.from_numpy(train_currquat).float().to(self.device),
            'velcmd': torch.from_numpy(train_velcmd).float().to(self.device),
            'trajlength': train_trajlength
        }
        
        self.val_data = {
            'ims': torch.from_numpy(val_ims).float().to(self.device),
            'desvel': torch.from_numpy(val_desvel).float().to(self.device),
            'currquat': torch.from_numpy(val_currquat).float().to(self.device),
            'velcmd': torch.from_numpy(val_velcmd).float().to(self.device),
            'trajlength': val_trajlength
        }
        
        # 计算训练和验证步骤数
        self.num_train_steps = len(self.train_data['trajlength'])
        self.num_val_steps = len(self.val_data['trajlength'])
        
        # 轨迹起始索引
        self.train_traj_starts = np.cumsum(self.train_data['trajlength']) - self.train_data['trajlength']
        self.val_traj_starts = np.cumsum(self.val_data['trajlength']) - self.val_data['trajlength']
        
        print(f"[数据] 训练轨迹: {self.num_train_steps}, 验证轨迹: {self.num_val_steps}")
        print(f"[数据] 训练图像: {len(self.train_data['ims'])}, 验证图像: {len(self.val_data['ims'])}")
        
    def setup_model(self):
        """设置模型"""
        print(f"[模型] 初始化模型: {self.config['model_type']}, 分支: {self.config['branch']}")
        
        # 根据分支选择模型
        branch = self.config['branch']
        
        if branch == 'B':
            # MambaVision + SSM分支
            from experiments.mamba_branches.branch_B_mambavision_ssm.models.mambavision_ssm_model import MambaVisionSSMModel
            self.model = MambaVisionSSMModel().to(self.device)
            
        elif branch == 'C':
            # CNN + Mamba3分支
            from experiments.mamba_branches.branch_C_cnn_mamba3.models.cnn_mamba3_model import CNNMamba3Model
            self.model = CNNMamba3Model().to(self.device)
            
        elif branch == 'D':
            # STH + Mamba分支
            from experiments.mamba_branches.branch_D_sth_mamba.models.sth_mamba_model import STHMambaModel
            self.model = STHMambaModel().to(self.device)
            
        elif branch == 'E':
            # Decision Mamba分支
            from experiments.mamba_branches.branch_E_decisionmamba.models.decision_mamba_model import DecisionMambaModel
            self.model = DecisionMambaModel().to(self.device)
            
        else:
            # 默认使用DroneMamba
            self.model = model_library.DroneMamba().to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[模型] 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        # 从检查点加载
        if self.config['load_checkpoint'] and self.config['checkpoint_path']:
            print(f"[模型] 从检查点加载: {self.config['checkpoint_path']}")
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.config['checkpoint_path'],
                self.model,
                self.optimizer,
                self.scheduler
            )
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
            print(f"[模型] 恢复训练: epoch={self.current_epoch}, best_val_loss={self.best_val_loss:.6f}")
        
    def setup_optimizer(self):
        """设置优化器和调度器"""
        print(f"[优化器] 创建优化器: {self.config['optimizer']}, lr={self.config['lr']}")
        
        # 创建优化器
        self.optimizer = create_optimizer(self.model, self.config)
        
        # 创建学习率调度器
        total_training_steps = self.config['N_eps'] * self.num_train_steps
        self.scheduler = LearningRateScheduler(self.config, total_training_steps)
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        # 打乱训练轨迹顺序
        shuffled_indices = np.random.permutation(self.num_train_steps)
        train_traj_starts = self.train_traj_starts[shuffled_indices]
        train_traj_lengths = self.train_data['trajlength'][shuffled_indices]
        
        for step in range(self.num_train_steps):
            step_start_time = time.time()
            
            # 获取当前轨迹数据
            traj_idx = shuffled_indices[step]
            start_idx = train_traj_starts[step]
            traj_len = train_traj_lengths[step]
            
            # 准备输入数据
            traj_input = self.train_data['ims'][start_idx+1:start_idx+traj_len, :, :].unsqueeze(1)
            desvel = self.train_data['desvel'][start_idx+1:start_idx+traj_len].view(-1, 1)
            currquat = self.train_data['currquat'][start_idx+1:start_idx+traj_len]
            cmd_target = self.train_data['velcmd'][start_idx+1:start_idx+traj_len, :]
            
            # 归一化目标
            cmd_norm = cmd_target / desvel
            
            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model([traj_input, desvel, currquat])
            
            # 计算损失
            loss, loss_dict = self.loss_fn.compute_loss(pred, cmd_norm, self.model)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_norm = self.gradient_manager.clip_gradients(self.model)
            
            # 优化步骤
            self.optimizer.step()
            
            # 更新学习率
            current_lr = self.scheduler.step(self.optimizer)
            
            # 记录损失
            epoch_loss += loss.item()
            self.total_steps += 1
            
            # 记录到TensorBoard
            if self.total_steps % self.config['log_freq'] == 0:
                self.monitor.log_training_step(
                    self.total_steps,
                    loss.item(),
                    loss_dict,
                    self.optimizer
                )
                self.monitor.log_scalar('train/grad_norm', grad_norm.item(), self.total_steps)
                self.monitor.log_scalar('train/step_time', time.time() - step_start_time, self.total_steps)
            
            # 打印进度
            if (step + 1) % max(1, self.num_train_steps // 10) == 0:
                progress = (step + 1) / self.num_train_steps * 100
                print(f"[训练] Epoch {epoch+1}/{self.config['N_eps']} - {progress:.1f}% "
                      f"- Loss: {loss.item():.6f} - LR: {current_lr:.2e}")
        
        # 计算平均损失
        avg_loss = epoch_loss / self.num_train_steps
        epoch_time = time.time() - epoch_start_time
        
        # 记录epoch统计
        self.monitor.log_scalar('train/epoch_loss', avg_loss, epoch)
        self.monitor.log_scalar('train/epoch_time', epoch_time, epoch)
        
        # 记录梯度统计
        grad_stats = self.gradient_manager.get_gradient_stats()
        for stat_name, stat_value in grad_stats.items():
            self.monitor.log_scalar(f'train/{stat_name}', stat_value, epoch)
        
        return avg_loss, epoch_time
    
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {}
        val_start_time = time.time()
        
        with torch.no_grad():
            for step in range(self.num_val_steps):
                # 获取当前轨迹数据
                start_idx = self.val_traj_starts[step]
                traj_len = self.val_data['trajlength'][step]
                
                # 准备输入数据
                traj_input = self.val_data['ims'][start_idx+1:start_idx+traj_len, :, :].unsqueeze(1)
                desvel = self.val_data['desvel'][start_idx+1:start_idx+traj_len].view(-1, 1)
                currquat = self.val_data['currquat'][start_idx+1:start_idx+traj_len]
                cmd_target = self.val_data['velcmd'][start_idx+1:start_idx+traj_len, :]
                
                # 归一化目标
                cmd_norm = cmd_target / desvel
                
                # 前向传播
                pred = self.model([traj_input, desvel, currquat])
                
                # 计算损失
                loss, loss_dict = self.loss_fn.compute_loss(pred, cmd_norm)
                
                # 计算指标
                if step == 0:  # 只在第一个批次计算详细指标
                    metrics = self.loss_fn.compute_validation_metrics(pred, cmd_norm)
                    val_metrics.update(metrics)
                
                val_loss += loss.item()
        
        # 计算平均损失
        avg_loss = val_loss / self.num_val_steps
        val_time = time.time() - val_start_time
        
        # 记录验证结果
        self.monitor.log_validation_step(epoch, avg_loss, val_metrics)
        
        # 检查是否是最佳模型
        is_best = avg_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_loss
            print(f"[验证] 新的最佳模型! Val Loss: {avg_loss:.6f} (之前: {self.best_val_loss:.6f})")
        
        return avg_loss, val_metrics, val_time, is_best
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        metrics = {
            'val_loss': val_loss,
            'epoch': epoch,
            'total_steps': self.total_steps,
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            epoch,
            self.model,
            self.optimizer,
            self.scheduler,
            metrics,
            is_best
        )
        
        print(f"[检查点] 保存到: {checkpoint_path}")
        return checkpoint_path
    
    def train(self):
        """主训练循环"""
        print(f"[训练] 开始训练，共 {self.config['N_eps']} 个epoch")
        print(f"[训练] 工作空间: {self.workspace}")
        print(f"[训练] 设备: {self.device}")
        
        # 设置数据
        self.setup_data()
        
        # 设置模型
        self.setup_model()
        
        # 设置优化器
        self.setup_optimizer()
        
        # 记录模型图
        if self.config['tensorboard']:
            try:
                # 使用一个样本记录模型图
                sample_input = [
                    torch.randn(1, 1, 60, 90, device=self.device),
                    torch.randn(1, 1, device=self.device),
                    torch.randn(1, 4, device=self.device)
                ]
                self.monitor.log_model_graph(self.model, sample_input)
            except Exception as e:
                print(f"[警告] 无法记录模型图: {e}")
        
        # 训练循环
        total_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['N_eps']):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"[训练] Epoch {epoch+1}/{self.config['N_eps']}")
            print(f"{'='*60}")
            
            # 训练一个epoch
            train_loss, train_time = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.config['val_freq'] == 0:
                val_loss, val_metrics, val_time, is_best = self.validate(epoch)
                
                # 打印验证结果
                print(f"[验证] Loss: {val_loss:.6f}, Time: {val_time:.2f}s")
                for metric_name, metric_value in val_metrics.items():
                    print(f"[验证] {metric_name}: {metric_value:.6f}")
            else:
                val_loss = None
                is_best = False
            
            # 保存检查点
            if (epoch + 1) % self.config['save_model_freq'] == 0 or epoch == self.config['N_eps'] - 1:
                self.save_checkpoint(epoch, val_loss or train_loss, is_best)
            
            # 打印epoch总结
            epoch_time = time.time() - epoch_start_time
            print(f"[总结] Epoch {epoch+1} - Train Loss: {train_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s ({train_time:.2f}s train, {val_time if val_loss else 0:.2f}s val)")
            
            # 刷新监控器
            self.monitor.flush()
        
        # 训练完成
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"[训练] 完成! 总时间: {total_time:.2f}s")
        print(f"[训练] 最佳验证损失: {self.best_val_loss:.6f}")
        print(f"[训练] 工作空间: {self.workspace}")
        print(f"{'='*60}")
        
        # 保存最终模型
        final_checkpoint = self.save_checkpoint(
            self.config['N_eps'] - 1,
            self.best_val_loss,
            False
        )
        
        # 保存训练历史
        self.monitor.save_history()
        
        # 关闭监控器
        self.monitor.close()
        
        # 关闭日志文件
        self.log_file.close()
        
        return final_checkpoint
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()
        
        if hasattr(self, 'monitor'):
            self.monitor.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Mamba分支标准化训练脚本')
    
    # 配置文件参数
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (.txt, .json, .yaml)')
    
    # 覆盖配置参数
    parser.add_argument('--device', type=str, help='训练设备 (cuda/cpu)')
    parser.add_argument('--branch', type=str, choices=['B', 'C', 'D', 'E'],
                       help='Mamba分支 (B, C, D, E)')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--epochs', type=int, help='训练epoch数')
    parser.add_argument('--batch_size', type=int, help='批大小')
    parser.add_argument('--experiment_name', type=str, help='实验名称')
    
    args = parser.parse_args()
    
    # 收集覆盖参数
    override_kwargs = {}
    if args.device:
        override_kwargs['device'] = args.device
    if args.branch:
        override_kwargs['branch'] = args.branch
    if args.lr:
        override_kwargs['lr'] = args.lr
    if args.epochs:
        override_kwargs['N_eps'] = args.epochs
    if args.batch_size:
        override_kwargs['batch_size'] = args.batch_size
    if args.experiment_name:
        override_kwargs['experiment_name'] = args.experiment_name
    
    try:
        # 创建训练器
        trainer = MambaBranchTrainer(args.config, **override_kwargs)
        
        # 开始训练
        final_checkpoint = trainer.train()
        
        print(f"\n训练完成! 最终模型保存在: {final_checkpoint}")
        print(f"使用以下命令启动TensorBoard:")
        print(f"  tensorboard --logdir {trainer.workspace / 'tensorboard'}")
        
        return 0
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # 确保清理
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == '__main__':
    sys.exit(main())