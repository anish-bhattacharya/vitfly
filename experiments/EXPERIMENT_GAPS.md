# 实验缺口追踪 — Training Pipeline

> 供训练管线查阅：以下实验已有评测代码就绪，需训练后即可仿真验证。

---

## P0: G_basic (CNN+MLP) — 最关键的对照实验

**位置**: `experiments/mamba_branches/branch_G_cnn_baseline/models/cnn_baseline_model.py`
**注册**: `train_mamba_optimized.py` — branch name `G`
**架构**: E的CNN编码器(0.46M) + 2层MLP头(0.16M) = 0.62M, 0.74ms
**用途**: 回答"SSM时序头是否必要？"——魔鬼代言人最核心质疑
**启动**:
```bash
python train_mamba_optimized.py --branches G --epochs 100
```
训练完成后运行仿真：
```bash
# 从envtest/ros/运行
bash run_full_test.bash G DecisionMamba bc 5.0
```

**预期**:
- 若 G_basic 达 ≤2次碰撞 → SSM头基本多余，叙事需重构
- 若 G_basic 远差于E蒸馏(>4次) → SSM头确有贡献

---

## P0: G_lstm (CNN+LSTM) — 延迟对比

**位置**: 同上文件，`CNNLSTMNet`
**注册**: `train_mamba_optimized.py` — branch name `G_lstm`
**架构**: E的CNN编码器(0.46M) + 单层LSTM(128) = 0.79M, 1.00ms
**用途**: 在相同编码器下对比LSTM vs SSM头的延迟和性能，验证LSTM归因修正

---

## P1: 多种子统计复现训练

**种子**: 43, 44, 45 (已有部分checkpoint)
**用处**: 补充均值±标准差，回应评审关于统计严谨性的质疑

| 模型 | 种子 | 训练脚本 | 状态 |
|------|------|---------|------|
| E蒸馏 | 43/44/45 | `train_distill.py` | ✅ 已有checkpoint |
| B+蒸馏 | 43/44/45 | `train_distill.py` | ✅ 已有checkpoint |
| E BC | 43/44/45 | `train_mamba_optimized.py` | ⏳ 已有种子43 |

---

## P1: CNN-线性基线 (CNNBase)

**位置**: `experiments/mamba_branches/branch_E_decisionmamba/models/cnn_linear_baseline.py`
**注册**: `train_mamba_optimized.py` — branch name `CNNBase`
**架构**: E的CNN编码器(0.46M) + MLP头(1.8M) = 2.26M
**配置**: `training/configs/cnn_linear_baseline.yaml`
**启动**:
```bash
python train_mamba_optimized.py --branches CNNBase --config training/configs/cnn_linear_baseline.yaml
```

---

## P2: 7m/s完整测试矩阵

| 架构 | 5m/s BC | 5m/s蒸馏 | 7m/s BC | 7m/s蒸馏 |
|------|:-------:|:--------:|:--------:|:--------:|
| A | ✅ 3 | ✅ 3 | ⏳ | ⏳ |
| B | DNF | ✅ 2 | ⏳ | ⏳ |
| B+ | ✅ 3 | ✅ 1 | ⏳ | ⏳ |
| C | ✅ 3 | ✅ 3 | ⏳ | ⏳ |
| D | ✅ 2 | ✅ 2 | ⏳ | ⏳ |
| E | ✅ 3 | ✅ 1 | ⏳ | ⏳ |

所有7m/s测试已可在仿真管线中一键运行: `bash run_pr_evaluations.bash`

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `experiments/mamba_branches/branch_G_cnn_baseline/models/cnn_baseline_model.py` | G_basic/G_lstm模型定义 |
| `training/train_mamba_optimized.py` | 训练入口(已注册G/CNNBase) |
| `training/configs/cnn_linear_baseline.yaml` | CNNBase训练配置 |
| `envtest/ros/run_competition.py` | 评测入口(已注册所有新分支) |
| `envtest/ros/user_code.py` | 模型分发(已注册CNNBaselineNet/CNNLinearBaselineNet) |
| `run_pr_evaluations.bash` | 批量仿真评测脚本 |
