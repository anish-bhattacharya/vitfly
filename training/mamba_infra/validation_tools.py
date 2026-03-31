"""
Mamba训练验证和测试工具
提供训练过程验证、性能评估和结果分析功能
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    step_time: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    

@dataclass
class ModelPerformance:
    """模型性能评估"""
    model_name: str
    branch: str
    parameters: int
    inference_time: float  # 毫秒
    memory_usage: float  # MB
    val_loss: float
    test_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    

class TrainingValidator:
    """训练验证器"""
    
    def __init__(self, workspace_dir: str):
        """
        初始化训练验证器
        
        Args:
            workspace_dir: 训练工作空间目录
        """
        self.workspace = Path(workspace_dir)
        self.metrics_file = self.workspace / 'training_history.json'
        self.config_file = self.workspace / 'config.yaml'
        
        # 加载数据
        self.metrics = self._load_metrics()
        self.config = self._load_config()
        
    def _load_metrics(self) -> List[TrainingMetrics]:
        """加载训练指标"""
        if not self.metrics_file.exists():
            return []
            
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                
            metrics = []
            for epoch_data in data.get('train_metrics', []):
                metric = TrainingMetrics(
                    epoch=epoch_data.get('epoch', 0),
                    train_loss=epoch_data.get('train_loss', 0),
                    val_loss=epoch_data.get('val_loss'),
                    learning_rate=epoch_data.get('learning_rate'),
                    grad_norm=epoch_data.get('grad_norm'),
                    step_time=epoch_data.get('step_time'),
                    timestamp=epoch_data.get('timestamp', '')
                )
                metrics.append(metric)
                
            return metrics
            
        except Exception as e:
            print(f"加载指标失败: {e}")
            return []
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config_files = [
            self.workspace / 'config.yaml',
            self.workspace / 'config.json',
            self.workspace / 'config.txt'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    if config_file.suffix == '.yaml':
                        with open(config_file, 'r') as f:
                            return yaml.safe_load(f)
                    elif config_file.suffix == '.json':
                        with open(config_file, 'r') as f:
                            return json.load(f)
                    elif config_file.suffix == '.txt':
                        return self._load_txt_config(config_file)
                except Exception as e:
                    print(f"加载配置失败 {config_file}: {e}")
                    
        return {}
    
    def _load_txt_config(self, config_path: Path) -> Dict[str, Any]:
        """加载TXT格式配置"""
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试解析值
                    try:
                        if value.startswith('[') and value.endswith(']'):
                            value = json.loads(value)
                        elif value.replace('.', '').replace('-', '').isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                    except:
                        pass
                    
                    config[key] = value
                    
        return config
    
    def validate_training_progress(self) -> Dict[str, Any]:
        """验证训练进度"""
        if not self.metrics:
            return {'status': 'error', 'message': '没有训练指标数据'}
        
        # 分析训练指标
        train_losses = [m.train_loss for m in self.metrics]
        val_losses = [m.val_loss for m in self.metrics if m.val_loss is not None]
        
        analysis = {
            'status': 'healthy',
            'total_epochs': len(self.metrics),
            'validation_epochs': len(val_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'issues': []
        }
        
        # 检查训练损失
        if train_losses:
            # 检查损失是否下降
            if len(train_losses) > 10:
                early_avg = np.mean(train_losses[:5])
                late_avg = np.mean(train_losses[-5:])
                if late_avg > early_avg * 1.5:
                    analysis['issues'].append('训练损失后期上升')
                    analysis['status'] = 'warning'
            
            # 检查损失是否稳定
            if len(train_losses) > 20:
                last_10 = train_losses[-10:]
                std_last_10 = np.std(last_10)
                if std_last_10 > np.mean(last_10) * 0.5:
                    analysis['issues'].append('训练损失波动较大')
                    analysis['status'] = 'warning'
        
        # 检查验证损失
        if val_losses:
            # 检查过拟合
            if len(val_losses) > 5:
                train_last = train_losses[-1] if train_losses else None
                val_last = val_losses[-1]
                if train_last and val_last > train_last * 2.0:
                    analysis['issues'].append('可能过拟合 (验证损失远高于训练损失)')
                    analysis['status'] = 'warning'
            
            # 检查验证损失趋势
            if len(val_losses) > 10:
                val_diff = val_losses[-1] - val_losses[0]
                if val_diff > 0:
                    analysis['issues'].append('验证损失整体上升')
                    analysis['status'] = 'warning'
        
        return analysis
    
    def generate_training_report(self, output_dir: Optional[str] = None) -> Path:
        """生成训练报告"""
        if output_dir is None:
            output_dir = self.workspace / 'reports'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成报告文件
        report_path = output_dir / 'training_report.md'
        
        # 验证训练进度
        validation = self.validate_training_progress()
        
        # 收集数据
        train_losses = [m.train_loss for m in self.metrics]
        val_losses = [m.val_loss for m in self.metrics if m.val_loss is not None]
        learning_rates = [m.learning_rate for m in self.metrics if m.learning_rate is not None]
        grad_norms = [m.grad_norm for m in self.metrics if m.grad_norm is not None]
        
        # 生成Markdown报告
        report_content = f"""# 训练报告

## 基本信息
- **工作空间**: {self.workspace.name}
- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总epoch数**: {len(self.metrics)}
- **验证epoch数**: {len(val_losses)}

## 训练状态
- **状态**: {validation['status']}
- **最终训练损失**: {validation['final_train_loss']:.6f}
- **最终验证损失**: {validation['final_val_loss']:.6f}

## 配置信息
```
{json.dumps(self.config, indent=2, ensure_ascii=False)}
```

## 问题检测
"""
        
        if validation['issues']:
            for issue in validation['issues']:
                report_content += f"- ⚠️ {issue}\n"
        else:
            report_content += "- ✅ 未检测到明显问题\n"
        
        # 添加统计信息
        report_content += f"""
## 统计信息

### 训练损失
- 最小值: {min(train_losses):.6f}
- 最大值: {max(train_losses):.6f}
- 平均值: {np.mean(train_losses):.6f}
- 标准差: {np.std(train_losses):.6f}

### 验证损失
"""
        
        if val_losses:
            report_content += f"""- 最小值: {min(val_losses):.6f}
- 最大值: {max(val_losses):.6f}
- 平均值: {np.mean(val_losses):.6f}
- 标准差: {np.std(val_losses):.6f}
"""
        else:
            report_content += "- 无验证数据\n"
        
        # 添加建议
        report_content += """
## 建议

"""
        
        if validation['status'] == 'healthy':
            report_content += "✅ 训练进展良好，可以继续或完成训练。\n"
        elif validation['status'] == 'warning':
            report_content += "⚠️ 检测到一些问题，建议：\n"
            for issue in validation['issues']:
                if '学习率' in issue:
                    report_content += "  - 考虑降低学习率或增加预热\n"
                elif '梯度' in issue:
                    report_content += "  - 检查梯度裁剪设置\n"
                elif '过拟合' in issue:
                    report_content += "  - 增加正则化或使用早停\n"
                elif '波动' in issue:
                    report_content += "  - 检查数据质量或批大小\n"
        else:
            report_content += "❌ 训练存在问题，建议检查配置和数据。\n"
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 生成图表
        self._generate_training_plots(output_dir)
        
        print(f"生成训练报告: {report_path}")
        return report_path
    
    def _generate_training_plots(self, output_dir: Path):
        """生成训练图表"""
        # 准备数据
        epochs = list(range(1, len(self.metrics) + 1))
        train_losses = [m.train_loss for m in self.metrics]
        val_epochs = [i+1 for i, m in enumerate(self.metrics) if m.val_loss is not None]
        val_losses = [m.val_loss for m in self.metrics if m.val_loss is not None]
        learning_rates = [m.learning_rate for m in self.metrics if m.learning_rate is not None]
        grad_norms = [m.grad_norm for m in self.metrics if m.grad_norm is not None]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失图
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        if val_losses:
            ax1.plot(val_epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率图
        ax2 = axes[0, 1]
        if learning_rates:
            ax2.plot(epochs[:len(learning_rates)], learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('学习率')
            ax2.set_title('学习率变化')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # 梯度范数图
        ax3 = axes[1, 0]
        if grad_norms:
            ax3.plot(epochs[:len(grad_norms)], grad_norms, 'm-', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('梯度范数')
            ax3.set_title('梯度范数变化')
            ax3.grid(True, alpha=0.3)
        
        # 损失分布图
        ax4 = axes[1, 1]
        if len(train_losses) > 10:
            # 使用滑动窗口计算平均损失
            window_size = max(1, len(train_losses) // 10)
            smoothed_losses = pd.Series(train_losses).rolling(window=window_size).mean()
            ax4.plot(epochs, train_losses, 'b-', alpha=0.3, label='原始损失')
            ax4.plot(epochs, smoothed_losses, 'b-', linewidth=2, label=f'滑动平均 (窗口={window_size})')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('损失')
            ax4.set_title('训练损失平滑')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"生成训练图表: {plot_path}")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_checkpoint: str, device: str = 'cuda'):
        """
        初始化模型评估器
        
        Args:
            model_checkpoint: 模型检查点路径
            device: 评估设备
        """
        self.checkpoint_path = Path(model_checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        self.checkpoint = self._load_checkpoint()
        self.config = self.checkpoint.get('config', {})
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_checkpoint(self) -> Dict[str, Any]:
        """加载检查点"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {self.checkpoint_path}")
            
        return torch.load(self.checkpoint_path, map_location='cpu')
    
    def _load_model(self) -> torch.nn.Module:
        """加载模型"""
        # 根据分支选择模型
        branch = self.config.get('branch', 'B')
        
        try:
            if branch == 'B':
                from experiments.mamba_branches.branch_B_mambavision_ssm.models.mambavision_ssm_model import MambaVisionSSMModel
                model = MambaVisionSSMModel()
            elif branch == 'C':
                from experiments.mamba_branches.branch_C_cnn_mamba3.models.cnn_mamba3_model import CNNMamba3Model
                model = CNNMamba3Model()
            elif branch == 'D':
                from experiments.mamba_branches.branch_D_sth_mamba.models.sth_mamba_model import STHMambaModel
                model = STHMambaModel()
            elif branch == 'E':
                from experiments.mamba_branches.branch_E_decisionmamba.models.decision_mamba_model import DecisionMambaModel
                model = DecisionMambaModel()
            else:
                # 默认使用DroneMamba
                import sys
                sys.path.append(str(Path(__file__).parent.parent.parent / 'models'))
                import model as model_library
                model = model_library.DroneMamba()
            
            # 加载权重
            model.load_state_dict(self.checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def evaluate_inference_speed(self, 
                               input_shape: Tuple[int, int, int] = (1, 60, 90),
                               num_iterations: int = 100) -> Dict[str, float]:
        """评估推理速度"""
        # 准备测试输入
        batch_size = 1
        test_input = [
            torch.randn(batch_size, 1, *input_shape, device=self.device),
            torch.randn(batch_size, 1, device=self.device),
            torch.randn(batch_size, 4, device=self.device)
        ]
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(test_input)
        
        # 测量推理时间
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if self.device.type == 'cuda':
            start_time.record()
        else:
            import time
            start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(test_input)
        
        if self.device.type == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
            inference_time_ms = start_time.elapsed_time(end_time) / num_iterations
        else:
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000 / num_iterations
        
        # 测量内存使用
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        else:
            import psutil
            process = psutil.Process()
            memory_allocated = process.memory_info().rss / 1024 / 1024  # MB
            memory_reserved = memory_allocated
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'inference_time_ms': inference_time_ms,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'fps': 1000 / inference_time_ms if inference_time_ms > 0 else 0
        }
    
    def evaluate_on_dataset(self,
                          data_dir: str,
                          num_samples: int = 100) -> Dict[str, float]:
        """在数据集上评估模型"""
        # 导入数据加载器
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from dataloading import dataloader
        
        # 加载数据
        data = dataloader(
            data_dir=data_dir,
            val_split=0.0,  # 不使用验证分割
            short=0,
            seed=42
        )
        
        (train_ims, train_desvel, train_currquat, train_velcmd, train_trajlength,
         _, _, _, _, _) = data
        
        # 转换为Tensor
        ims = torch.from_numpy(train_ims).float().to(self.device)
        desvel = torch.from_numpy(train_desvel).float().to(self.device)
        currquat = torch.from_numpy(train_currquat).float().to(self.device)
        velcmd = torch.from_numpy(train_velcmd).float().to(self.device)
        
        # 随机选择样本
        num_total = min(len(ims), num_samples)
        indices = torch.randperm(len(ims))[:num_total]
        
        # 评估
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for idx in indices:
                # 准备输入
                traj_input = ims[idx:idx+1].unsqueeze(1)  # [1, 1, H, W]
                desvel_input = desvel[idx:idx+1].view(-1, 1)
                currquat_input = currquat[idx:idx+1]
                cmd_target = velcmd[idx:idx+1, :]
                
                # 前向传播
                pred = self.model([traj_input, desvel_input, currquat_input])
                
                # 计算损失
                cmd_norm = cmd_target / desvel_input
                loss = torch.nn.functional.mse_loss(pred, cmd_norm)
                
                total_loss += loss.item()
                predictions.append(pred.cpu().numpy())
                targets.append(cmd_norm.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / num_total
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # 计算其他指标
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'num_samples': num_total,
            'predictions_mean': float(np.mean(predictions)),
            'predictions_std': float(np.std(predictions)),
            'targets_mean': float(np.mean(targets)),
            'targets_std': float(np.std(targets))
        }
    
    def generate_evaluation_report(self,
                                 output_dir: Optional[str] = None,
                                 test_data_dir: Optional[str] = None) -> Path:
        """生成评估报告"""
        if output_dir is None:
            output_dir = self.checkpoint_path.parent / 'evaluation'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 评估推理速度
        print("评估推理速度...")
        speed_metrics = self.evaluate_inference_speed()
        
        # 评估模型性能
        print("评估模型性能...")
        if test_data_dir:
            perf_metrics = self.evaluate_on_dataset(test_data_dir)
        else:
            perf_metrics = {'loss': self.checkpoint.get('metrics', {}).get('val_loss', 0)}
        
        # 生成报告
        report_path = output_dir / 'evaluation_report.md'
        
        report_content = f"""# 模型评估报告

## 基本信息
- **模型检查点**: {self.checkpoint_path.name}
- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **设备**: {self.device}
- **分支**: {self.config.get('branch', '未知')}

## 模型信息
- **总参数量**: {speed_metrics['total_parameters']:,}
- **可训练参数量**: {speed_metrics['trainable_parameters']:,}
- **模型大小**: {speed_metrics['total_parameters'] * 4 / 1024 / 1024:.2f} MB (float32)

## 推理性能
- **推理时间**: {speed_metrics['inference_time_ms']:.2f} ms
- **帧率**: {speed_metrics['fps']:.1f} FPS
- **GPU内存使用**: {speed_metrics['memory_allocated_mb']:.1f} MB
- **GPU内存预留**: {speed_metrics['memory_reserved_mb']:.1f} MB

## 模型性能
"""
        
        if 'loss' in perf_metrics:
            report_content += f"- **损失**: {perf_metrics['loss']:.6f}\n"
        if 'mae' in perf_metrics:
            report_content += f"- **MAE**: {perf_metrics['mae']:.6f}\n"
        if 'rmse' in perf_metrics:
            report_content += f"- **RMSE**: {perf_metrics['rmse']:.6f}\n"
        if 'r2_score' in perf_metrics:
            report_content += f"- **R²分数**: {perf_metrics['r2_score']:.6f}\n"
        
        # 添加训练信息
        report_content += f"""
## 训练信息
- **训练epoch**: {self.checkpoint.get('epoch', '未知')}
- **最佳验证损失**: {self.checkpoint.get('metrics', {}).get('best_val_loss', '未知')}
- **总训练步数**: {self.checkpoint.get('metrics', {}).get('total_steps', '未知')}

## 配置信息
```
{json.dumps(self.config, indent=2, ensure_ascii=False)}
```

## 评估结果
"""
        
        # 判断模型性能
        inference_time = speed_metrics['inference_time_ms']
        if inference_time < 5:
            report_content += "✅ **推理速度优秀** (<5ms)\n"
        elif inference_time < 10:
            report_content += "⚠️ **推理速度良好** (5-10ms)\n"
        else:
            report_content += "❌ **推理速度较慢** (>10ms)\n"
        
        if 'loss' in perf_metrics:
            loss = perf_metrics['loss']
            if loss < 0.01:
                report_content += "✅ **模型精度优秀** (损失<0.01)\n"
            elif loss < 0.05:
                report_content += "⚠️ **模型精度良好** (损失0.01-0.05)\n"
            else:
                report_content += "❌ **模型精度需要改进** (损失>0.05)\n"
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存详细指标
        metrics_path = output_dir / 'evaluation_metrics.json'
        all_metrics = {
            'speed_metrics': speed_metrics,
            'performance_metrics': perf_metrics,
            'config': self.config,
            'checkpoint_info': {
                'epoch': self.checkpoint.get('epoch'),
                'metrics': self.checkpoint.get('metrics', {})
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"生成评估报告: {report_path}")
        return report_path


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str):
        """
        初始化结果分析器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        
    def analyze_multiple_experiments(self) -> pd.DataFrame:
        """分析多个实验"""
        # 查找所有评估结果
        evaluation_files = list(self.results_dir.glob('**/evaluation_metrics.json'))
        
        results = []
        for eval_file in evaluation_files:
            try:
                with open(eval_file, 'r') as f:
                    data = json.load(f)
                
                # 提取关键信息
                result = {
                    'experiment': eval_file.parent.parent.name,
                    'branch': data.get('config', {}).get('branch', 'unknown'),
                    'lr': data.get('config', {}).get('lr', 0),
                    'batch_size': data.get('config', {}).get('batch_size', 1),
                    'epochs': data.get('checkpoint_info', {}).get('epoch', 0),
                    'inference_time_ms': data.get('speed_metrics', {}).get('inference_time_ms', 0),
                    'fps': data.get('speed_metrics', {}).get('fps', 0),
                    'total_parameters': data.get('speed_metrics', {}).get('total_parameters', 0),
                    'val_loss': data.get('checkpoint_info', {}).get('metrics', {}).get('val_loss', 0),
                    'test_loss': data.get('performance_metrics', {}).get('loss', 0),
                    'mae': data.get('performance_metrics', {}).get('mae', 0),
                    'rmse': data.get('performance_metrics', {}).get('rmse', 0),
                    'r2_score': data.get('performance_metrics', {}).get('r2_score', 0),
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"分析结果失败 {eval_file}: {e}")
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        if not df.empty:
            # 保存分析结果
            output_path = self.results_dir / 'experiment_analysis.csv'
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            # 生成汇总报告
            self._generate_summary_report(df)
        
        return df
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """生成汇总报告"""
        if df.empty:
            return
        
        report_path = self.results_dir / 'experiment_summary.md'
        
        report_content = "# 实验汇总报告\n\n"
        report_content += f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"**实验数量**: {len(df)}\n\n"
        
        # 按分支分组
        if 'branch' in df.columns:
            report_content += "## 按分支统计\n\n"
            branch_stats = df.groupby('branch').agg({
                'val_loss': ['mean', 'min', 'max'],
                'inference_time_ms': ['mean', 'min', 'max'],
                'fps': ['mean', 'min', 'max'],
                'total_parameters': 'mean'
            }).round(4)
            
            report_content += branch_stats.to_markdown() + "\n\n"
        
        # 最佳模型
        report_content += "## 最佳模型\n\n"
        
        if 'val_loss' in df.columns:
            best_by_loss = df.loc[df['val_loss'].idxmin()]
            report_content += f"### 最佳验证损失\n"
            report_content += f"- **实验**: {best_by_loss['experiment']}\n"
            report_content += f"- **分支**: {best_by_loss['branch']}\n"
            report_content += f"- **验证损失**: {best_by_loss['val_loss']:.6f}\n"
            report_content += f"- **推理时间**: {best_by_loss['inference_time_ms']:.2f} ms\n\n"
        
        if 'inference_time_ms' in df.columns:
            fastest = df.loc[df['inference_time_ms'].idxmin()]
            report_content += f"### 最快推理速度\n"
            report_content += f"- **实验**: {fastest['experiment']}\n"
            report_content += f"- **分支**: {fastest['branch']}\n"
            report_content += f"- **推理时间**: {fastest['inference_time_ms']:.2f} ms\n"
            report_content += f"- **FPS**: {fastest['fps']:.1f}\n"
            report_content += f"- **验证损失**: {fastest['val_loss']:.6f}\n\n"
        
        # 相关性分析
        report_content += "## 相关性分析\n\n"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # 找出与验证损失相关性最强的特征
            if 'val_loss' in corr_matrix.columns:
                val_loss_corr = corr_matrix['val_loss'].abs().sort_values(ascending=False)
                report_content += "### 与验证损失的相关性\n"
                for feature, corr in val_loss_corr.items():
                    if feature != 'val_loss' and not pd.isna(corr):
                        report_content += f"- **{feature}**: {corr:.3f}\n"
                report_content += "\n"
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"生成汇总报告: {report_path}")


# 示例使用
if __name__ == '__main__':
    # 示例1: 验证训练进度
    print("示例1: 验证训练进度")
    validator = TrainingValidator('training/logs/mamba_experiment')
    validation_result = validator.validate_training_progress()
    print(f"验证结果: {validation_result}")
    
    # 生成训练报告
    report_path = validator.generate_training_report()
    print(f"训练报告: {report_path}")
    
    # 示例2: 评估模型
    print("\n示例2: 评估模型")
    evaluator = ModelEvaluator('training/logs/mamba_experiment/checkpoints/checkpoint_best.pt')
    
    # 评估推理速度
    speed_metrics = evaluator.evaluate_inference_speed()
    print(f"推理速度: {speed_metrics['inference_time_ms']:.2f} ms")
    print(f"FPS: {speed_metrics['fps']:.1f}")
    
    # 生成评估报告
    eval_report = evaluator.generate_evaluation_report()
    print(f"评估报告: {eval_report}")
    
    # 示例3: 分析多个实验
    print("\n示例3: 分析多个实验")
    analyzer = ResultAnalyzer('training/logs')
    results_df = analyzer.analyze_multiple_experiments()
    
    if not results_df.empty:
        print(f"分析 {len(results_df)} 个实验")
        print("最佳验证损失:")
        print(results_df.nsmallest(3, 'val_loss')[['experiment', 'branch', 'val_loss', 'inference_time_ms']])