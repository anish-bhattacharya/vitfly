# ViT-Fly Mamba分支共享训练基础设施

## 概述

本基础设施为ViT-Fly项目的Mamba分支(B-E)提供标准化的训练框架，支持：

- **统一训练接口**: 所有Mamba分支使用相同的训练脚本
- **配置驱动**: YAML/JSON/TXT多种配置格式支持
- **模块化设计**: 可复用的训练组件
- **完整监控**: TensorBoard集成和训练验证
- **性能评估**: 推理速度和模型精度评估
- **实验管理**: 超参数搜索和消融实验

## 目录结构

```
training/mamba_infra/
├── README.md                    # 本文档
├── mamba_training_utils.py      # 共享训练工具模块
├── train_mamba_branch.py        # 标准化训练脚本
├── config_manager.py            # 配置管理工具
├── validation_tools.py          # 验证和测试工具
└── config_templates/            # 配置模板
    ├── base_config.yaml         # 基础配置模板
    ├── branch_B_config.yaml     # 分支B配置模板
    ├── branch_C_config.yaml     # 分支C配置模板
    ├── branch_D_config.yaml     # 分支D配置模板
    └── branch_E_config.yaml     # 分支E配置模板
```

## 快速开始

### 1. 安装依赖

确保已安装以下依赖：
```bash
pip install torch torchvision tensorboard pyyaml pandas matplotlib seaborn
```

### 2. 训练Mamba分支

使用标准化训练脚本训练任意Mamba分支：

```bash
# 训练分支B (MambaVision + SSM)
python training/mamba_infra/train_mamba_branch.py \
  --config training/mamba_infra/config_templates/branch_B_config.yaml \
  --experiment_name my_branch_B_experiment

# 训练分支C (CNN + Mamba3)
python training/mamba_infra/train_mamba_branch.py \
  --config training/mamba_infra/config_templates/branch_C_config.yaml \
  --branch C \
  --lr 1e-4 \
  --epochs 100

# 从检查点恢复训练
python training/mamba_infra/train_mamba_branch.py \
  --config training/mamba_infra/config_templates/base_config.yaml \
  --load_checkpoint true \
  --checkpoint_path training/logs/previous_experiment/checkpoints/checkpoint_latest.pt
```

### 3. 监控训练进度

启动TensorBoard监控训练：
```bash
tensorboard --logdir training/logs
```

然后在浏览器中打开：http://localhost:6006

## 配置系统

### 配置格式支持

支持三种配置格式：
- **YAML** (推荐): 结构化配置，支持继承
- **JSON**: 标准JSON格式
- **TXT**: 兼容现有ViT-Fly格式

### 配置模板

提供了5个预定义配置模板：

| 模板 | 描述 | 参数量 | 推荐学习率 |
|------|------|--------|------------|
| `base_config.yaml` | 基础配置 | - | 1e-4 |
| `branch_B_config.yaml` | MambaVision + SSM | ~3.3M | 1.2e-4 |
| `branch_C_config.yaml` | CNN + Mamba3 | ~2.8M | 1.0e-4 |
| `branch_D_config.yaml` | STH + Mamba | ~3.1M | 1.1e-4 |
| `branch_E_config.yaml` | Decision Mamba | ~2.5M | 9e-5 |

### 创建自定义配置

```python
from mamba_infra.config_manager import ConfigManager

# 创建配置管理器
manager = ConfigManager()

# 基于模板创建配置
config_path = manager.create_config(
    template_name='branch_B_config',
    output_path='my_config.yaml',
    lr=1.5e-4,
    N_eps=200,
    experiment_name='custom_experiment'
)
```

## 训练管理

### 超参数搜索

```python
from mamba_infra.config_manager import ConfigManager, HyperparameterRange

manager = ConfigManager()

# 定义超参数范围
hyperparams = [
    HyperparameterRange('lr', [1e-4, 5e-4, 1e-3], '学习率'),
    HyperparameterRange('batch_size', [1, 2, 4], '批大小'),
    HyperparameterRange('weight_decay', [1e-5, 1e-4, 1e-3], '权重衰减')
]

# 生成超参数搜索配置
configs = manager.generate_hyperparameter_search(
    template_name='base_config',
    output_dir='configs/hyperparam_search',
    hyperparameters=hyperparams
)
```

### 消融实验

```python
# 创建消融实验配置
ablation_configs = manager.create_ablation_study(
    template_name='branch_C_config',
    output_dir='configs/ablation_study',
    ablation_params={
        'lr_decay': [True, False],
        'loss_type': ['mse', 'l1', 'huber'],
        'optimizer': ['adamw', 'adam', 'sgd']
    }
)
```

### 批量训练

```bash
# 生成批量训练脚本
python -c "
from mamba_infra.config_manager import ConfigManager
manager = ConfigManager()
import glob
configs = glob.glob('configs/hyperparam_search/*.yaml')
manager.generate_training_script(
    config_paths=configs[:5],
    output_script='scripts/batch_train.sh',
    gpu_ids=[0, 1]
)
"

# 运行批量训练
bash scripts/batch_train.sh
```

## 验证和评估

### 训练过程验证

```python
from mamba_infra.validation_tools import TrainingValidator

# 验证训练进度
validator = TrainingValidator('training/logs/my_experiment')
validation_result = validator.validate_training_progress()
print(f"训练状态: {validation_result['status']}")

# 生成训练报告
report_path = validator.generate_training_report()
```

### 模型评估

```python
from mamba_infra.validation_tools import ModelEvaluator

# 评估模型性能
evaluator = ModelEvaluator('training/logs/my_experiment/checkpoints/checkpoint_best.pt')

# 评估推理速度
speed_metrics = evaluator.evaluate_inference_speed()
print(f"推理时间: {speed_metrics['inference_time_ms']:.2f} ms")
print(f"FPS: {speed_metrics['fps']:.1f}")

# 在数据集上评估
perf_metrics = evaluator.evaluate_on_dataset('training/datasets/data', num_samples=100)
print(f"测试损失: {perf_metrics['loss']:.6f}")

# 生成评估报告
eval_report = evaluator.generate_evaluation_report()
```

### 结果分析

```python
from mamba_infra.validation_tools import ResultAnalyzer

# 分析多个实验
analyzer = ResultAnalyzer('training/logs')
results_df = analyzer.analyze_multiple_experiments()

# 查看最佳模型
print("最佳验证损失:")
print(results_df.nsmallest(3, 'val_loss')[['experiment', 'branch', 'val_loss', 'inference_time_ms']])
```

## 核心组件

### 1. MambaTrainingConfig

配置管理类，支持多种格式的配置加载和保存。

```python
from mamba_infra.mamba_training_utils import MambaTrainingConfig

# 加载配置
config = MambaTrainingConfig('my_config.yaml')

# 访问配置参数
print(f"学习率: {config['lr']}")
print(f"训练epoch: {config['N_eps']}")

# 修改配置
config['lr'] = 2e-4
config.save_config('updated_config.yaml')
```

### 2. MambaLossFunction

标准化的损失函数，支持MSE、L1、Huber损失和L2正则化。

```python
from mamba_infra.mamba_training_utils import MambaLossFunction

loss_fn = MambaLossFunction(config)
loss, loss_dict = loss_fn.compute_loss(predictions, targets, model)
```

### 3. CheckpointManager

检查点管理，支持自动保存和加载。

```python
from mamba_infra.mamba_training_utils import CheckpointManager

checkpoint_manager = CheckpointManager(workspace, config)

# 保存检查点
checkpoint_path = checkpoint_manager.save_checkpoint(
    epoch=10,
    model=model,
    optimizer=optimizer,
    metrics={'val_loss': 0.05},
    is_best=True
)

# 加载检查点
checkpoint = checkpoint_manager.load_checkpoint(
    checkpoint_path,
    model,
    optimizer
)
```

### 4. TrainingMonitor

TensorBoard集成和训练监控。

```python
from mamba_infra.mamba_training_utils import TrainingMonitor

monitor = TrainingMonitor(workspace, config)

# 记录训练指标
monitor.log_training_step(step=100, loss=0.1, metrics={'mae': 0.05}, optimizer=optimizer)
monitor.log_validation_step(epoch=10, loss=0.08, metrics={'val_mae': 0.04})

# 保存训练历史
monitor.save_history()
```

## 与现有框架集成

### 数据兼容性

使用现有的`dataloading.py`模块，确保数据格式一致：

```python
from training.dataloading import dataloader

# 加载数据 (与现有代码兼容)
data = dataloader(
    data_dir='training/datasets/data',
    val_split=0.2,
    short=0,
    seed=42
)
```

### 模型兼容性

支持所有Mamba分支模型：

```python
# 分支B: MambaVision + SSM
from experiments.mamba_branches.branch_B_mambavision_ssm.models.mambavision_ssm_model import MambaVisionSSMModel

# 分支C: CNN + Mamba3
from experiments.mamba_branches.branch_C_cnn_mamba3.models.cnn_mamba3_model import CNNMamba3Model

# 分支D: STH + Mamba
from experiments.mamba_branches.branch_D_sth_mamba.models.sth_mamba_model import STHMambaModel

# 分支E: Decision Mamba
from experiments.mamba_branches.branch_E_decisionmamba.models.decision_mamba_model import DecisionMambaModel

# 默认: DroneMamba
from models.model import DroneMamba
```

## 最佳实践

### 1. 训练配置

- **学习率**: 从1e-4开始，根据模型大小调整
- **批大小**: 使用轨迹级别批处理 (batch_size=1)
- **正则化**: 根据模型复杂度调整权重衰减
- **验证频率**: 每10个epoch验证一次
- **保存频率**: 每25个epoch保存一次检查点

### 2. 监控建议

- 定期检查TensorBoard中的训练/验证损失曲线
- 监控梯度范数，避免梯度爆炸/消失
- 检查学习率变化是否符合预期
- 使用训练验证器检测潜在问题

### 3. 性能优化

- 使用混合精度训练加速训练
- 启用CUDA Graph优化推理速度
- 使用梯度累积模拟更大批大小
- 合理设置检查点保存频率，避免IO瓶颈

## 故障排除

### 常见问题

1. **训练损失不下降**
   - 检查学习率是否合适
   - 验证数据加载是否正确
   - 检查模型初始化

2. **验证损失远高于训练损失**
   - 可能过拟合，增加正则化
   - 检查验证集数据质量
   - 调整早停策略

3. **GPU内存不足**
   - 减少批大小
   - 使用梯度累积
   - 启用混合精度训练

4. **训练速度慢**
   - 检查数据加载瓶颈
   - 启用CUDA优化
   - 使用更快的存储设备

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据形状
print(f"输入形状: {inputs.shape}")
print(f"目标形状: {targets.shape}")

# 检查模型参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## 扩展开发

### 添加新的Mamba分支

1. 在`experiments/mamba_branches/`中创建新分支目录
2. 实现模型类，确保接口兼容
3. 创建对应的配置模板
4. 在`train_mamba_branch.py`中添加分支支持

### 自定义损失函数

```python
from mamba_infra.mamba_training_utils import MambaLossFunction

class CustomLossFunction(MambaLossFunction):
    def compute_loss(self, predictions, targets, model=None):
        # 自定义损失计算
        base_loss = super().compute_loss(predictions, targets, model)
        # 添加自定义损失项
        custom_loss = self.compute_custom_term(predictions, targets)
        return base_loss + custom_loss
```

### 添加新的监控指标

```python
from mamba_infra.mamba_training_utils import TrainingMonitor

class EnhancedMonitor(TrainingMonitor):
    def log_custom_metric(self, tag, value, step):
        self.log_scalar(f'custom/{tag}', value, step)
        # 添加到历史记录
        if 'custom_metrics' not in self.history:
            self.history['custom_metrics'] = []
        self.history['custom_metrics'].append({'tag': tag, 'value': value, 'step': step})
```

## 许可证

本项目基于ViT-Fly项目的许可证。详见项目根目录的LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进本基础设施。

## 联系方式

如有问题，请参考ViT-Fly项目文档或提交Issue。