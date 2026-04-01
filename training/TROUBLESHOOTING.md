# Training Troubleshooting Guide

本文档记录了本项目开发过程中发现的常见问题及解决方案。

## 问题索引

| 问题 | 症状 | 严重程度 |
|------|------|----------|
| 验证集为空 | Val Loss = `inf` | 🔴 Critical |
| 数据不匹配 | "Number of images and telemetry still do not match" | 🟡 Warning |
| Git Push 失败 | "RPC failed", "Error in the HTTP2 framing layer" | 🟡 Warning |
| 模型不收敛 | Train loss 不下降 | 🟡 Warning |

---

## 1. 验证集为空 (Val Loss = inf)

### 症状
```
Epoch   1/1 | Train Loss: 0.1234 | Val Loss: inf | LR: 0.000100 | Time: 1.5s
```

### 原因
当使用小数据集（如 <10 条轨迹）时，trajectory-level 的验证集划分会导致验证集为空。

### 解决方案

#### 方案A: 使用 sample-level 划分（推荐）
在 `dataloading.py` 中确保使用 sample-level split：

```python
# 检查代码是否有此逻辑：
if num_val_trajs == 0 and val_split > 0 and len(traj_lengths) >= 2:
    # 使用 sample-level split
    num_val_samples = int(val_split * len(traj_meta_full))
    ...
```

#### 方案B: 增加数据量
使用更多轨迹：
```bash
python train_mamba_optimized.py --short 50  # 至少 10+ 条轨迹
```

#### 方案C: 调整 val_split
```bash
python train_mamba_optimized.py --val_split 0.3  # 增加验证集比例
```

### 验证命令
```bash
python train_mamba_optimized.py --branches A --epochs 1 --short 20 --val_split 0.2
```
确认输出包含：
```
Training samples: X
Validation samples: Y  # 必须 > 0
```

---

## 2. 数据不匹配警告

### 症状
```
[DATALOADER] Number of images and telemetry still do not match in 170743211265, skipping
[DATALOADER] Extra image found at end of data, cutting it from 170700241053
```

### 原因
某些轨迹的图像数量与元数据行数不一致。

### 解决方案
这是**正常行为**。dataloader 会自动：
- 跳过不匹配的轨迹
- 裁剪多余的图像

不影响训练。

---

## 3. Git Push 失败

### 症状
```
error: RPC failed; curl 16 Error in the HTTP2 framing layer
fatal: the remote end hung up unexpectedly
```

### 解决方案

#### 方案A: 重试（等待网络恢复）
```bash
git push origin setup-evaluation-20260321
```

#### 方案B: 使用 gh CLI
```bash
# 安装 gh
apt-get install -y gh

# 认证
echo "ghp_YOUR_TOKEN" | gh auth login --with-token

# 推送
git push https://github.com/Liber1917/vitfly.git branch-name
```

#### 方案C: 使用 embeded token
```bash
git push https://USER:TOKEN@github.com/REPO.git branch-name
```

---

## 4. 模型不收敛

### 症状
Train loss 长时间不下降，或波动很大。

### 排查步骤

1. **检查数据是否正确加载**
```python
# 验证 target 是 3D velocity，不是重复的标量
# 错误: target = [desired_vels[idx]] * 3  # 错误！
# 正确: target = velocity.clone()
```

2. **检查学习率**
```bash
# 尝试降低学习率
python train_mamba_optimized.py --lr 0.00005
```

3. **增加训练数据**
```bash
# 使用更多轨迹
python train_mamba_optimized.py --short 200
```

4. **检查 GPU 可用性**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. 训练脚本快速验证

在提交任何更改前，运行以下验证：

```bash
cd /root/vitfly/training

# 1. 验证 import
python -c "import train_mamba_optimized; import dataloading; print('OK')"

# 2. 验证数据加载
python -c "
from dataloading import dataloader
train_data, val_data, _, _ = dataloader('/root/vitfly/training/datasets/data_full', val_split=0.2, short=10)
print(f'Train: {len(train_data[1])}, Val: {len(val_data[1])}')
"

# 3. 验证训练（1 epoch）
python train_mamba_optimized.py --branches A --epochs 1 --short 10 --val_split 0.2
```

预期输出：
- Import: 无错误
- Data: Validation samples > 0
- Training: Val Loss 是有限数值（非 inf）

---

## 6. 调试技巧

### 查看 GPU 使用
```bash
nvidia-smi
# 或在训练脚本中查看（每50个batch自动打印）
```

### 查看详细日志
```bash
python train_mamba_optimized.py --branches A --epochs 1 --short 5 2>&1 | tee training.log
```

### 检查模型参数数量
```python
from vmamba_lstm_model import create_vmamba_lstm_model
model = create_vmamba_lstm_model()
print(sum(p.numel() for p in model.parameters()))
```

---

## 7. 已知修复记录

| 日期 | 问题 | 修复 |
|------|------|------|
| 2026-04-01 | Target 使用重复标量而非3D velocity | 修改 dataset `__getitem__` 使用 `velocity.clone()` |
| 2026-04-01 | 空验证集 | 在 dataloading.py 添加 sample-level split 逻辑 |
| 2026-04-01 | Git push 超时 | 使用 gh CLI 或 embed token |

---

## 8. 紧急回滚

如果训练脚本出错，可以使用 Git 回滚：

```bash
# 查看历史
git log --oneline -10

# 回滚到上一个正常版本
git checkout HEAD~1 -- training/train_mamba_optimized.py
git checkout HEAD~1 -- training/dataloading.py
```