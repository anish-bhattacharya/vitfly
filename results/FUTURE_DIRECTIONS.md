# Future Directions: Beyond Behavior Cloning

A brainstorm based on literature review (May 2026).
Revised with 60m simulation results (May 2026).

## Hard Constraints

- **Flightmare/Unity simulation**: Cannot be modified (fixed environment)
- **Expert data**: Pre-collected offline dataset, no online expert queries possible
- **Simulation pipeline**: Separately managed, cannot integrate RL rollouts
- **Available**: 253 valid trajectories (42,156 images) + 327 skipped trajectories (62,920 images)
- **Data**: Pre-recorded depth images + telemetry + expert velocity commands at 25-30Hz

Any proposed method must work within these constraints.

## Feasibility Matrix (Updated)

| # | Approach | Within Constraints? | Effort | Expected Gain | Status |
|---|----------|-------------------|--------|---------------|--------|
| A | Multi-step sequence prediction | ✅ Yes | Low | Medium | 🟡 已在跑（MPS并行seq=4,8,16） |
| B | Data augmentation | ✅ Yes | Low | Medium | ⏳ 待做 |
| C | Knowledge distillation (ViT+LSTM → Mamba) | ✅ Yes | Low | High | ✅ **完成：B+/E (1 crash) > 教师 (2 crash)** |
| C2 | Born-again distillation (Mamba → Mamba) | ✅ Yes | Low | High | ⏳ 待做 |
| C3 | Distillation loss weight tuning | ✅ Yes | Low | Medium | ⏳ 待做 |
| D | Pseudo-label 62K skipped images | ✅ Yes | Medium | High | ⏳ 待做 |
| E | Hard example mining | ✅ Yes | Low | Medium | ⏳ 待做 |
| F | Ensemble inference | ✅ Yes | Low | Low | ⏳ 待做 |
| G | DAgger / online data | ❌ | High | High | ❌ 不可行 |
| H | RL fine-tuning | ❌ | Very High | Very High | ❌ 不可行 |
| I | New SSM architecture design | ✅ Yes | Medium | Potentially 0 crash | ⏳ 构思中 |

## Feasible Directions

### A. Multi-Step Sequence Prediction
**Status**: 🟡 **MPS并行训练中（Branch E, seq_len=4/8/16, 100ep）**。
已完成单帧蒸馏实验，当前正在用NVIDIA MPS并行训练多步BC模型。
预计完成时间：~2小时。完成后需进行蒸馏+序列训练的联合实验。

### B. Data Augmentation
**Status**: ⏳ 待实现。保持不变。

### C. Cross-Architecture Knowledge Distillation (ViT+LSTM → Mamba)
**Status**: ✅ **全部完成**。

**60m全量测试关键结果**：
- B+ (MambaVision+Mamba-3): BC 3 crash → **蒸馏 1 crash** 🏆（超越教师）
- E (DecisionMamba, 纯SSM): BC 3 crash → **蒸馏 1 crash** 🏆（超越教师）
- B (MambaVision+SSM): BC Failed → **蒸馏 2 crash** ✅（被蒸馏救活）
- A/C/D: 蒸馏持平BC
- 教师ViT+LSTM (3.56M) 基准: **2 crash**
- **所有6分支：蒸馏从未损害性能**

**SSM纯度与蒸馏效果的关系（关键发现）**：

| 分支 | 视觉编码器 | 时序头 | SSM纯度 | 蒸馏结果 |
|------|-----------|--------|---------|---------|
| E | SSM (Coarse+Fine) | SSM | **纯SSM** | 1 crash 🏆 |
| B+ | 混合(Attention+SSM) | SSM (Mamba-3) | **两端有SSM** | 1 crash 🏆 |
| B | 混合(Attention+SSM) | SSM | 两端有SSM | 2 crash |
| D | CNN-like | SSM (Mamba-2) | 仅时序SSM | 2 crash |
| A | SSM (SS2D) | **LSTM** | 仅视觉SSM | 3 crash |
| C | **CNN** | SSM (Mamba-3) | 仅时序SSM | 3 crash |

**核心规律**：SSM必须在视觉和时序两端同时存在，跨架构蒸馏才能最有效。

### C2. Born-Again Iterative Distillation (Mamba → Mamba)
**Status**: ⏳ 待实现。

用蒸馏冠军（B+ 或 E）作为新教师，再次蒸馏其他Mamba分支。
假设：同架构蒸馏比跨架构更高效，Mamba→Mamba的知识传递损失更小。

```python
teacher = load_distilled_best()  # B+ 或 E (1 crash)
student = create_mamba_model(branch)
loss = alpha * feat_align + beta * distill + gamma * gt
```

`train_distill.py` 已支持 `--teacher-branch` 参数，仅需指定教师检查点路径。

### C3. Distillation Loss Weight Optimization
**Status**: ⏳ 待做。

当前α=β=γ=1.0是随手选的。需要进行网格搜索确定最优权重。

搜索空间：
- α (feature alignment): {0.0, 0.5, 1.0, 2.0}
- β (output distill): {0.0, 0.5, 1.0, 2.0}
- γ (GT loss): {0.5, 1.0, 2.0}
- 推荐先在 Branch E 上跑网格搜索（~9组），确定合理范围

### D. Pseudo-Label 62K Skipped Images
**Status**: ⏳ 待实现。保持不变。

### E. Hard Example Mining
**Status**: ⏳ 待实现。保持不变。

### F. Ensemble Inference
**Status**: ⏳ 待测试。保持不变。

## G. New Architecture: MambaFusion (SOTA目标: 0 crash)

**Status**: ⏳ **构思中，基于SSM纯度实验证据**。

### 动机

现有最佳蒸馏模型(B+, E)各1 crash。根据SSM纯度分析，这1 crash可能来自：

1. **架构层面**：当前无模型是 "纯SSM视觉 + 最先进SSM时序" 的组合
2. **训练层面**：蒸馏权重未优化，序列训练未启用
3. **测试层面**：不确定1 crash的位置——是否是所有模型共通的难点

### 设计方案

```
MambaFusion — 双路径SSM架构

路径1: 全局SSM (MambaVision混合编码器)
  输入: 60×90 depth → Stem + Stage1-3
  产出: 512-dim 全局特征 (障碍物布局)

路径2: 局部SSM (DecisionMamba FineSSM)  
  输入: 60×90 depth → CNNEmbed → FineSSM
  产出: 256-dim 局部特征 (近距障碍细节)

融合: 可学习的门控加权
  feat = gate * global + (1-gate) * local
  gate = σ(W_1 * global + W_2 * local)

时序: Mamba-3 Head
  输入: feat + velocity + quat
  处理: 2层Mamba-3 SSM
  输出: (vx, vy, vz)
```

### 验证路径

0 crash不一定要靠新架构。推荐顺序：

```
Step 1: 等MPS seq_len=16训练完成 → 测试E在序列模式下是否0 crash
Step 2: 如否，在E上调优α,β,γ（网格搜索）
Step 3: 如仍否，设计MambaFusion
Step 4: 蒸馏训练MambaFusion
Step 5: 60m仿真验证
```

## Recommended Order (Revised)

1. ✅ **C (distillation)** → 已完成，B+/E (1 crash) > 教师 (2 crash)
2. 🟡 **A (multi-step)** → MPS并行训练中
3. ⏳ **C3 (loss weight tuning)** → 先在E上扫α,β,γ
4. ⏳ **C2 (born-again distill)** → B+/E为教师
5. ⏳ **B (data augmentation)** → 数据增强
6. ⏳ **G (MambaFusion architecture)** → 如上述步骤未达0 crash

## Training & Simulation Status

✅ **蒸馏训练完成**：6分支全部50epoch蒸馏，收敛正常。
✅ **仿真验证完成**：60m全量测试，含教师基线。
🟡 **序列训练中**：Branch E, seq_len=4/8/16, MPS并行, ~2h完成。

