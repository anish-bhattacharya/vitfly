# DroneMamba 训练指南

> **警告**：本文档记录了 DroneMamba 模型的训练流程、关键配置和常见问题。请在开始训练前仔细阅读，避免重复踩坑。

## 概述

DroneMamba 是基于 Mamba (State Space Model, SSM) 的四旋翼无人机避障模型，相比 ViT 架构具有参数量少、推理速度快的优势。

| 模型 | 参数量 | 推理速度 (CPU) | 相对 ViT |
|-----|--------:|--------------:|---------:|
| ViT | 3,101K | 5.89ms | 1.00x |
| ViT+LSTM | 3,563K | 7.62ms | 0.77x |
| **DroneMamba (SSM)** | **452K** | **2.21ms** | **2.66x** ✅ |
| **DroneMamba (LSTM)** | **635K** | **2.70ms** | **2.19x** ✅ |

---

## 训练流程

### 步骤 1：准备数据集

确保训练数据集已准备好：

```bash
ls /home/vitfly/training/datasets/data/
# 应该看到多个以时间戳命名的轨迹文件夹
```

每个轨迹文件夹应包含：
- `*.png` - 深度图像序列
- `data.csv` - 速度命令和遥测数据

### 步骤 2：选择配置文件

DroneMamba 有两种变体：

| 配置 | 时序建模 | 推荐场景 |
|-----|---------|---------|
| `train_mamba.txt` | SSM (纯 Mamba) | 实验性，参数量最少 |
| 自定义配置 | LSTM (混合) | 更成熟，推荐首次使用 |

**推荐配置**（LSTM 版本）：

```txt
device = cuda
basedir = /home/vitfly
logdir = training/logs
datadir = training/datasets

dataset = data
short = 0
val_split = 0.2

model_type = DroneMamba_LSTM
load_checkpoint = False
checkpoint_path = ''

lr = 1e-4
N_eps = 100
lr_warmup_epochs = 5
lr_decay = False
save_model_freq = 25
val_freq = 10
```

### 步骤 3：启动训练

```bash
cd /home/vitfly/training
python3 train.py --config config/train_mamba.txt
```

**或使用自定义参数**：

```bash
python3 train.py --config config/train_mamba.txt \
    --model_type DroneMamba_LSTM \
    --lr 1e-4 \
    --N_eps 100 \
    --batch_size 32
```

### 步骤 4：监控训练

使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir /home/vitfly/training/logs
```

---

## 关键配置项

### 模型类型 (`model_type`)

```txt
model_type = DroneMamba        # 纯 SSM 版本（实验性）
model_type = DroneMamba_LSTM   # LSTM 版本（推荐）
```

### 学习率 (`lr`)

- 推荐范围：`1e-4` 到 `1e-3`
- 使用学习率预热 (`lr_warmup_epochs = 5`)

### 训练轮数 (`N_eps`)

- 最小：50 epochs
- 推荐：100 epochs
- 完整训练：125+ epochs

### 验证集比例 (`val_split`)

- 默认：`0.2` (20% 数据用于验证)

### 保存频率 (`save_model_freq`)

- 每 N 个 epoch 保存一次模型
- 推荐：10-25 epochs

---

## 训练日志解读

### 正常训练日志

```
[LearnerLSTM init] Making workspace /home/vitfly/training/logs/d03_22_t01_01
[DATALOADER] Loading from training/datasets/data
[DATALOADER] Dataloading done | train images (70109, 60, 90), val images (14999, 60, 90)
[DATALOADER] Preloading into device cuda done
[SETUP] Establishing model and optimizer.
[TRAIN] Completed epoch 1/100, ep_loss = 0.021234, time = 10.36s
[VAL] Validating for val set of size 14999 images
[VAL] Completed validation, val_loss = 0.019876, time taken = 0.55s
[SAVE] Saving model at epoch 50
```

### 损失值范围

| 阶段 | 正常损失范围 |
|-----|-------------|
| 初始 (epoch 1-10) | 0.02 - 0.05 |
| 中期 (epoch 10-50) | 0.015 - 0.025 |
| 后期 (epoch 50+) | 0.015 - 0.020 |

### 训练速度

- 每 epoch 约 10-15 秒（GPU）
- 100 epochs 约 20-25 分钟

---

## 常见问题及解决方案

### 1. CUDA Out of Memory

**问题**：
```
RuntimeError: CUDA out of memory.
```

**解决方案**：
- 减小 `batch_size`（默认 32 → 16 或 8）
- 使用 `short = 1` 加载部分数据集进行测试

### 2. 训练损失不下降

**可能原因**：
- 学习率过低
- 数据集质量差
- 模型加载了错误的 checkpoint

**解决方案**：
```txt
# 提高学习率
lr = 1e-3

# 从头开始训练
load_checkpoint = False
```

### 3. 验证损失高于训练损失

**可能原因**：
- 过拟合
- 验证集分布不同

**解决方案**：
- 启用学习率衰减：`lr_decay = True`
- 增加验证频率：`val_freq = 5`

### 4. 模型加载失败

**问题**：
```
RuntimeError: Error(s) in loading state_dict
```

**原因**：模型架构不匹配（SSM vs LSTM）

**解决方案**：
- 确保 `model_type` 与 checkpoint 一致
- 或设置 `load_checkpoint = False` 从头训练

---

## 模型架构

### DroneMamba (LSTM 版本) - 推荐

```
输入：深度图像 (B, 1, 60, 90)
     期望速度 (B, 1)
     四元数 (B, 4)
       ↓
Stage 1: CNN 特征提取 (2 层卷积)
       ↓
Stage 2: Mamba Encoder (2 层 SSM Block)
       ↓
Stage 3: 特征融合与解码
       ↓
Stage 4: LSTM 时序建模
       ↓
输出：速度命令 (B, 3) [vx, vy, vz]
```

### DroneMamba (SSM 版本) - 实验性

与 LSTM 版本唯一区别：
- Stage 4 使用 Simplified-SSM 代替 LSTM

**参数量对比**：
- LSTM 版本：635K
- SSM 版本：452K (-29%)

---

## 检查点管理

### 模型保存位置

```
/home/vitfly/training/logs/<实验名称>/
├── model_000050.pth    # epoch 50
├── model_000100.pth    # epoch 100
├── log.txt             # 训练日志
├── args.txt            # 参数配置
└── events.out.tfevents.*  # TensorBoard 日志
```

### 恢复训练

```bash
python3 train.py --config config/train_mamba.txt \
    --load_checkpoint True \
    --checkpoint_path /home/vitfly/training/logs/d03_22_t01_01/model_000100.pth
```

### 部署模型

复制最新模型到 checkpoints 目录：

```bash
cp /home/vitfly/training/logs/<最新实验>/model_000*.pth \
   /home/vitfly/checkpoints/drone_mamba_latest.pth
```

---

## 评估流程

### 步骤 1：设置环境变量

```bash
cd /home/vitfly
export MODEL_TYPE="DroneMamba"
export MODEL_PATH="../../checkpoints/drone_mamba_latest.pth"
export RENDER=1  # 启用 Unity 渲染
```

### 步骤 2：运行评估

```bash
bash launch_evaluation.bash 1 vision
```

### 步骤 3：查看结果

```bash
cat evaluation.yaml
```

**结果示例**：
```yaml
rollout_1:
  Success: false
  number_crashes: 6
  time_to_finish: 9.27
```

---

## 性能优化建议

### 1. 超参数调优

| 参数 | 推荐搜索范围 |
|-----|-------------|
| `lr` | [1e-5, 1e-4, 1e-3] |
| `hidden_size` | [64, 128, 256] |
| `d_state` | [4, 8, 16] |
| `N_eps` | [50, 100, 150] |

### 2. 数据增强

当前数据集有限（580 轨迹），建议：
- 收集更多仿真数据
- 使用不同环境（trees, spheres_hard）

### 3. 混合精度训练

修改 `train.py` 启用 AMP：

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 训练循环中
with autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
```

---

## 文件清单

### 核心文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 训练脚本 | `training/train.py` | 主训练循环 |
| 模型定义 | `models/model.py` | DroneMamba 类 |
| SSM 模块 | `models/mamba_submodules.py` | SSM 核心组件 |
| 配置文件 | `training/config/train_mamba.txt` | 训练配置 |

### 辅助脚本

| 文件 | 用途 |
|------|------|
| `models/test_mamba.py` | 前向传播测试 |
| `models/test_mamba_grad.py` | 梯度流动测试 |
| `models/benchmark_mamba.py` | 推理速度基准 |

---

## 调试命令

```bash
# 检查 GPU 状态
nvidia-smi

# 查看训练日志
tail -f /home/vitfly/training/logs/<实验名称>/log.txt

# 检查模型文件
ls -lh /home/vitfly/training/logs/<实验名称>/*.pth

# TensorBoard 监控
tensorboard --logdir /home/vitfly/training/logs --port 6006
```

---

## 已知限制

1. **Python 循环瓶颈**：SSM 光栅扫描使用 Python 循环，推理速度受限
   - 未来优化：使用 CUDA kernel 或 Triton

2. **数据集规模**：当前数据集较小（580 轨迹）
   - 建议：使用 `bash launch_evaluation.bash 10 state` 收集更多数据

3. **超参数未优化**：当前配置基于经验值
   - 建议：进行网格搜索

---

## 成功训练的标志

- ✅ 训练损失稳定下降到 0.015-0.020
- ✅ 验证损失与训练损失接近
- ✅ 模型文件大小约 1.8MB
- ✅ 仿真评估能发布速度命令
- ✅ 无人机能够起飞并向前飞行

---

## 参考资源

- [ViT-Fly 原始论文](https://arxiv.org/abs/2405.10391)
- [Mamba 论文](https://arxiv.org/abs/2312.00752)
- [DroneMamba 实施总结](MAMBA_IMPLEMENTATION_SUMMARY.md)
- [调试总结](DEBUGGING_SUMMARY.md)

---

*最后更新：2026-03-22*