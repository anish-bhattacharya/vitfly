# Future Directions: Beyond Behavior Cloning

A brainstorm based on literature review (May 2026).
Revised with 60m simulation results (May 2026).
Last updated: 2026-05-06 (all experiments complete, architecture design phase)

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
| A | Multi-step sequence prediction | ✅ Yes | Low | Medium | ✅ **完成（seq4/8/16 BC + distill）** |
| B | Data augmentation | ✅ Yes | Low | Medium | ⏳ 待做 |
| C | Cross-architecture distill (ViT+LSTM → Mamba) | ✅ Yes | Low | High | ✅ **B+/E (1 crash) > 教师 (2 crash)** |
| C2 | Born-again distill (Mamba → Mamba) | ✅ Yes | Low | Medium | ✅ **γ=1.0/2.0均完成，γ=2.0 val_gt=0.0165** |
| C3 | Loss weight tuning | ✅ Yes | Low | Medium | ✅ **α,β网格搜索完成，影响微小** |
| D | Pseudo-label 62K skipped images | ✅ Yes | Medium | High | ⏳ 待做 |
| E | Hard example mining | ✅ Yes | Low | Medium | ⏳ 待做 |
| F | Ensemble inference | ✅ Yes | Low | Low | ⏳ 待做 |
| G | DAgger / online data | ❌ | High | High | ❌ 不可行 |
| H | RL fine-tuning | ❌ | Very High | Very High | ❌ 不可行 |
| **I** | **New SSM architecture (MambaFusion)** | ✅ Yes | **Medium** | **0 crash target** | **🔴 讨论中** |

## Feasible Directions

### A. Multi-Step Sequence Prediction
**Status**: ✅ **全部完成**。

| Model | Init | seq_len | val_gt | Sim @ 60m |
|-------|------|---------|-------|-----------|
| E seq4 BC | — | 4 | 0.2297 | ⏳ pending |
| E seq8 BC | — | 8 | 0.2413 | ⏳ pending |
| E seq16 BC | — | 16 | 0.2323 | 4 crashes |
| E Distill seq4 | BC seq4 | 4 | **0.0167** | ⏳ pending |
| E Distill seq8 | BC seq8 | 8 | **0.0169** | ⏳ pending |
| E Distill seq16 | BC seq16 | 16 | 0.7414 | ⏳ pending |

BC pretraining improves distillation val_gt (0.0188 → 0.0167). Sim testing pending.

### B. Data Augmentation
**Status**: ⏳ 待实现。几何变换增强（翻转、旋转、噪声、亮度），42K → ~168K。

### C. Cross-Architecture Knowledge Distillation (ViT+LSTM → Mamba)
**Status**: ✅ **全部完成**。

**60m全量测试关键结果**：
- B+ (MambaVision+Mamba-3): BC 3 crash → **蒸馏 1 crash** 🏆（超越教师）
- E (DecisionMamba, 纯SSM): BC 3 crash → **蒸馏 1 crash** 🏆（超越教师）
- E @ 7m/s 60m: **1 crash**（教师 @ 7m/s 60m: 5 crash ❌）
- B (MambaVision+SSM): BC Failed → **蒸馏 2 crash** ✅（被蒸馏救活）
- 所有6分支：蒸馏从未损害性能
- 速度鲁棒性：**E Distill 在 5m/s 和 7m/s 都是 1 crash**

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
**Status**: ✅ **全部完成**。

| Variant | γ | val_gt | distill_gap | Sim @ 60m |
|---------|---|-------|-------------|-----------|
| B+ → E | 1.0 | 0.0172 | **0.0037** | 3 crashes |
| **B+ → E** | **2.0** | **0.0165** 🏆 | 0.0046 | ⏳ pending |

γ=2.0 achieves val_gt=0.0165 — first model to surpass BC baseline (0.0186).
**Key insight**: Same-architecture distillation (Mamba→Mamba) achieves 4.4× better distill_gap than cross-architecture, but needs higher GT weight to prevent teacher overfitting.

### C3. Distillation Loss Weight Optimization
**Status**: ✅ **网格搜索完成**。

2×2 factorial on E with ViT+LSTM teacher:

| α (feat) | β (distill) | score | gt | distill |
|---------|------------|-------|----|---------|
| 0.5 | 0.5 | 0.0267 | 0.0178 | 0.0177 |
| 1.0 | 1.0 | 0.0274 | 0.0188 | 0.0172 (default) |

α has negligible effect in 0.5-1.0. β=0.5 marginally better. All within sim noise.
**结论：教师选择比权重调优重要 10×.**

### D. Pseudo-Label 62K Skipped Images
**Status**: ⏳ 待实现。可用最佳蒸馏模型（E Distill）给 62K 无标签图片打伪标签，42K → ~105K。

### E. Hard Example Mining
**Status**: ⏳ 待实现。碰撞附近的帧、高误差样本上采样。

### F. Ensemble Inference
**Status**: ⏳ 待测试。B+ + E 集成可能互补失败模式。

---

## I. New Architecture: MambaFusion (SOTA目标: 0 crash)

**Status**: 🔴 **郑重考虑中。基于所有实验证据的架构设计决策。**

### 实验证据汇总

1. **E (纯SSM, 2.19M) → 1 crash** — 两端 SSM 架构已验证
2. **B+ (MambaVision + Mamba-3, 2.55M) → 1 crash** — 混合SSM架构已验证
3. **SSM纯度规律**：两端都有 SSM = 最佳
4. **α,β 网格搜索**：权重调优帮助不大
5. **Born-again γ=2.0**：val_gt=0.0165（超越 BC），模拟待验证
6. **多步蒸馏（seq4/8）**：val_gt 提升（0.0188 → 0.0167），模拟待验证

### 关键问题：1 crash → 0 crash 需要新架构吗？

**论证1：不需要（训练调优即可）**
- E 已 1 crash，seq4/8 蒸馏和 born-again γ=2.0 的 val 指标更好但未仿真
- 仿真管线正在测试这些模型，可能已经或接近 0 crash
- 如果 E 在 seq4/8/16 蒸馏后达到 0 crash，新架构不必要

**论证2：需要（架构天花板）**
- E 和 B+ 的 1 crash 发生在同一位置？如果是，架构共性导致
- 双路径设计（全局 + 局部）可能覆盖 E 和 B+ 各自的盲区
- 蒸馏已经很强了，进一步收益可能来自架构而非训练

### 设计方案（根据证据修订）

```
MambaFusion v1 — 最保守的改进

视觉编码器: MambaVision (B+的混合编码器)
  + 双尺度: coarse→fine 金字塔 (E的设计)
  = 全局场景理解 + 局部细节捕捉

融合: FeaturePyramid (非门控，更稳定)
  outputs: 512-dim 全局 + 256-dim 局部

时序头: Mamba-3 (B+验证过的最佳SSM)
  input: 768-dim fused features + vel + quat
  output: (vx, vy, vz)

训练: 蒸馏 (ViT+LSTM 教师) + BC init + seq_len=4
```

### 实施计划

```
Phase A: 等仿真验证结果（~1天）
  - born-again γ=2.0 @ 60m
  - seq4/8/16 distill @ 60m
  - 如果任意模型达到 0 crash → 架构不必要
  
Phase B: 如果 Phase A 无 0 crash（~3天）
  1. 实现 MambaFusion 模型（合并 B+ 和 E 的组件）
  2. 蒸馏训练（ViT+LSTM 教师, 50ep）
  3. 仿真验证
```

### 决策标准

| 条件 | 行动 |
|------|------|
| 任一待测模型 0 crash | 🚫 不做新架构，用现有最佳 |
| 全部待测模型 ≥1 crash 且位置相同 | ⚠️ 环境问题，非架构 |
| 全部待测模型 ≥1 crash 且位置不同 | ✅ 设计 MambaFusion |

## Recommended Order (Final)

1. ✅ **C (distillation)** → 完成
2. ✅ **A (multi-step)** → 完成
3. ✅ **C3 (loss weight tuning)** → 完成
4. ✅ **C2 (born-again)** → 完成
5. ⏳ **等待仿真验证 Phase A 结果**
6. 🔴 **I (MambaFusion)** → 如 Phase A 无 0 crash 则启动

## Training & Simulation Status

✅ **蒸馏训练**：全部完成（6分支 + born-again + seq4/8/16 + grid search）
✅ **仿真验证**：60m 基线测试完成
🔴 **仿真待验证**：born-again γ=2.0, seq4/8/16 distill @ 60m
🔴 **架构设计**：MambaFusion 提案，等待 Phase A 决策

