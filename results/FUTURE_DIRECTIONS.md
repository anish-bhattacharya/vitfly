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

### C. Knowledge Distillation (Branch B → Others)
**Status**: ⏳ 待实现。
Branch B passes simulation ✅. Use it as teacher:
```python
teacher = load_model('B').eval()
with torch.no_grad():
    soft_target = teacher(depth, vel, quat)
hard_target = expert_velcmd  # original ground truth
loss = alpha * mse(student_out, soft_target) \
     + (1-alpha) * mse(student_out, hard_target)
```
Alpha can be annealed from 1.0 → 0.0 during training (start with imitation of the working model, gradually shift to ground truth).

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
**Status**: ⏳ 待测试。
Average predictions across multiple branches:
```python
v_final = (v_B + v_Bplus + v_C + v_D + v_E) / 5
```
Simple, zero-cost improvement in inference stability.

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
- A: 重训中 (Epoch 5/100, Val Loss 0.0219, 目标<0.0194)
