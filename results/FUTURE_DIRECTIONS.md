# Future Directions: Beyond Behavior Cloning

A brainstorm based on literature review (May 2026).
Revised with project constraints.

## Hard Constraints

- **Flightmare/Unity simulation**: Cannot be modified (fixed environment)
- **Expert data**: Pre-collected offline dataset, no online expert queries possible
- **Simulation pipeline**: Separately managed, cannot integrate RL rollouts
- **Available**: 253 valid trajectories (42,156 images) + 327 skipped trajectories (62,920 images)
- **Data**: Pre-recorded depth images + telemetry + expert velocity commands at 25-30Hz

Any proposed method must work within these constraints.

## Feasibility Matrix

| # | Approach | Within Constraints? | Effort | Expected Gain | Status |
|---|----------|-------------------|--------|---------------|--------|
| A | Multi-step sequence prediction (`--sequence_length N`) | ✅ Yes | Low | Medium | ✅ **完成** |
| B | Data augmentation (rotation, flip, noise, brightness) | ✅ Yes | Low | Medium | ⏳ 待做 |
| C | Knowledge distillation (B → A/C/D/E/B+) | ✅ Yes | Low | High | ⏳ 待做 |
| D | Utilize 62K skipped images via pseudo-labeling | ✅ Yes | Medium | High | ⏳ 待做 |
| E | Hard example mining / sample reweighting | ✅ Yes | Low | Medium | ⏳ 待做 |
| F | Ensemble inference (weighted voting) | ✅ Yes | Low | Low | ⏳ 待做 |
| G | DAgger (online data collection) | ❌ Needs simulator interaction | High | High | ❌ 不可行 |
| H | RL fine-tuning | ❌ Needs RL env + reward | Very High | Very High | ❌ 不可行 |
| I | Curriculum terrain generation | ❌ Needs environment modification | Very High | High | ❌ 不可行 |
| J | APC data augmentation | ❌ Needs online expert queries | Medium | High | ❌ 不可行 |

## Feasible Directions (A-F)

### A. Multi-Step Sequence Prediction
**Status**: ✅ **完成**。`--sequence_length N` 已实现并集成到训练管道。
消融实验表明 seq_len=16×100epoch（每帧损失0.0112）优于单帧基线（0.0194）。
在Branch D（Mamba-2 SSM）上验证，长序列收益更显著。
详见 `results/EXPERIMENT_REPORT.md §5.4`。

### B. Data Augmentation (No Expert Needed)
**Status**: ⏳ 待实现。
Augment the 42K image dataset with:
- Horizontal/vertical flip (left-right obstacle mirroring)
- Random rotation (±5°)
- Gaussian noise (simulate sensor noise)
- Brightness/contrast jitter (simulate lighting variation)
- Random crop + resize

All operations are **geometric/image transforms only** — they don't change the expert command.
A depth image flipped horizontally should still map to the same velocity target if the scene is symmetric.
For asymmetric scenes, this creates useful adversarial examples.

### C. Cross-Architecture Knowledge Distillation (ViT+LSTM → Mamba)
**Status**: ⏳ 待实现。

**文献依据**：
- MOHAWK (NeurIPS 2024): 三阶段蒸馏 Transformer → Mamba-2，3B token即有效
- CAB (2025): 跨架构注意力桥 Transformer Q,K ↔ Mamba B,C
- X-Distill (ICLR 2026): DINOv2 ViT → ResNet 用于机器人控制
- TransMamba (2025): 视觉Mamba的多方向扫描蒸馏

**关键结论**：naive蒸馏（只匹配输出）在跨架构时失败（MOHAWK证明），需要多阶段对齐。

**设计**：用上游ViT+LSTM（best model, 7m/s实飞）作为教师，蒸馏到所有Mamba分支：

```python
teacher = load_ViTLSTM().eval()  # 上游最佳模型
student = create_mamba_model(branch)  # B/C/D/E/B+/A

# 阶段1: 特征对齐（参考MOHAWK的stage 1-2）
loss_feat = MSE(student_encoder_features, teacher_encoder_features)

# 阶段2: 端到端蒸馏（参考MOHAWK的stage 3）
loss_distill = MSE(student_out, teacher_out)
loss_gt = MSE(student_out, ground_truth)
loss = alpha * loss_distill + beta * loss_feat + gamma * loss_gt
```

**科学问题**：
1. 跨架构（ViT→Mamba）知识能否有效转移？
2. 6种不同Mamba架构的蒸馏效率对比——什么架构吸收最好？
3. 蒸馏后的Mamba vs 纯BC的Mamba，仿真性能对比
4. 首次：跨架构蒸馏在机器人控制领域的系统性研究

**预期价值**：高。这是NLP/视觉之外跨架构蒸馏的首次机器人应用。

### D. Pseudo-Label 62K Skipped Images
**Status**: ⏳ 待实现。
The 327 skipped trajectories (62,920 images) have valid PNGs but CSV row count mismatches.
The best current model can generate pseudo-labels for these images:
```python
for img in skipped_images:
    pred = best_model.predict(img)
    # Add to training set with confidence weighting
    dataset.append((img, pred))
```
This nearly triples the dataset size (42K → 105K) at zero additional simulation cost.

### E. Hard Example Mining
**Status**: ⏳ 待实现。
Not all 42K samples are equally valuable. Identify hard examples by:
- High prediction error (model uncertainty)
- Collision-adjacent frames (near-miss events)
- Obstacle-dense scenarios

These samples can be upweighted in the loss function or oversampled.

### F. Ensemble Inference
**Status**: ⏳ 待测试，但需注意部署限制。

Average predictions across multiple branches:
```python
v_final = (v_B + v_Bplus + v_C + v_D + v_E) / 5
```
Simple, zero-cost improvement in inference stability.

**⚠️ 部署限制**：上游部署环境为CPU-only（Intel NUC 10, i7, 16GB RAM），单模型推理即需25ms。5模型集成推理需125ms（8Hz），**不满足30Hz控制频率要求**。

**替代方案**：将集成转化为**知识蒸馏（方向C）**——用5个teacher模型生成软标签，训练一个单一学生模型，推理延迟25ms不变，但吸收了多分支的集体智慧。

## Recommended Order

1. **B (data augmentation)** → Immediate, no code changes needed
2. **A (multi-step)** → ✅ **已完成**
3. **D (pseudo-label)** → Data multiplier
4. **C (distillation)** → Leverage Branch B's success
5. **E (hard example mining)** → Fine-tune on critical cases

## Training Status

✅ **所有6分支100-epoch全量训练完成。仿真验证5/6分支通过（A重训中）。**
- B: 仿真通过 (0 crash, 4.26s)
- B+: 仿真通过 (0 crash, 4.21s)
- C: 仿真通过 (0 crash, 4.20s)
- D: 仿真通过 (0 crash, 4.21s)
- E: 仿真通过 (0 crash, 4.21s)
- A: 重训中（后台agent验收中，待结果）
