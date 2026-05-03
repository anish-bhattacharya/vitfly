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

| # | Approach | Within Constraints? | Effort | Expected Gain |
|---|----------|-------------------|--------|---------------|
| A | Multi-step sequence prediction (`--sequence_length N`) | ✅ Yes | Low | Medium |
| B | Data augmentation (rotation, flip, noise, brightness) | ✅ Yes | Low | Medium |
| C | Knowledge distillation (B → A/C/D/E/B+) | ✅ Yes | Low | High |
| D | Utilize 62K skipped images via pseudo-labeling | ✅ Yes | Medium | High |
| E | Hard example mining / sample reweighting | ✅ Yes | Low | Medium |
| F | Ensemble inference (weighted voting) | ✅ Yes | Low | Low |
| G | DAgger (online data collection) | ❌ Needs simulator interaction | High | High |
| H | RL fine-tuning | ❌ Needs RL env + reward | Very High | Very High |
| I | Curriculum terrain generation | ❌ Needs environment modification | Very High | High |
| J | APC data augmentation | ❌ Needs online expert queries | Medium | High |

## Feasible Directions (A-F)

### A. Multi-Step Sequence Prediction
**Status**: Implemented (`--sequence_length N`).
Train with trajectory sequences (seq_len=8,16,32) instead of single frames.
Matches the upstream vitfly training approach (LSTM over trajectory segments).

### B. Data Augmentation (No Expert Needed)
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
Not all 42K samples are equally valuable. Identify hard examples by:
- High prediction error (model uncertainty)
- Collision-adjacent frames (near-miss events)
- Obstacle-dense scenarios

These samples can be upweighted in the loss function or oversampled.

### F. Ensemble Inference
Average predictions across multiple branches:
```python
v_final = (v_B + v_Bplus + v_C + v_D + v_E) / 5
```
Simple, zero-cost improvement in inference stability.

## Recommended Order

1. **B (data augmentation)** → Immediate, no code changes needed
2. **A (multi-step)** → Already implemented, just run experiment
3. **D (pseudo-label)** → Data multiplier
4. **C (distillation)** → Leverage Branch B's success
5. **E (hard example mining)** → Fine-tune on critical cases

## Training Status

⚠️ The previous full training (C/D/E/B+, 100 epochs with --compile) was killed.
Branch B (100 epochs) checkpoints are intact.
C/D/E/B+ need retraining. Use `--compile` flag.
