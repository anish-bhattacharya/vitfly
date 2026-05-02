# Training Collapse Analysis - Branch B Epoch 92

**Date**: 2026-05-02  
**Issue**: Retrained model (epoch=92) performs worse than undertrained model (epoch=1)  
**Root Cause**: Mode collapse due to imbalanced training labels

---

## Problem Summary

The retrained Branch B model (epoch=92, val_loss=0.009) crashes at 3.98s with 1 collision, while the old undertrained model (epoch=1, val_loss=0.019) successfully reaches 20m with 0 collisions.

**Paradox**: Lower validation loss → worse performance

---

## Root Cause: Mode Collapse

The model learned a **degenerate shortcut** that achieves low validation loss but fails in practice:

### Model Behavior Analysis

**Test**: Feed 20 random depth images with `vx_in=4.0 m/s`

**Expected**: Model outputs vary based on image content (obstacles, turns, etc.)

**Actual**: Model outputs are nearly constant regardless of image:
```
All 20 outputs: vx ≈ 4.000, vy ≈ -0.007, vz ≈ +0.07
vy variance: 0.0000042  (essentially zero)
vz variance: 0.0004     (essentially zero)
```

**Conclusion**: Model ignores the image and just echoes the forward velocity input.

### Input-Output Relationship

The model learned to copy `vx_in → vx_out` with near-zero lateral/vertical:

| vx_in | vx_out | Ratio | vy_out | vz_out |
|-------|--------|-------|--------|--------|
| 1.0 m/s | 1.178 m/s | 1.18 | -0.007 | +0.07 |
| 4.0 m/s | 4.069 m/s | 1.02 | -0.007 | +0.07 |
| 6.0 m/s | 6.029 m/s | 1.00 | -0.007 | +0.07 |

**Pattern**: Model outputs `[vx_in * ~1.0, ~0, ~0]` regardless of image content.

---

## Why This Happened

### Training Data Imbalance

The training data has **imbalanced labels**:
- Forward velocity (vx): Always present, varies 1-6 m/s
- Lateral velocity (vy): Mostly zero (expert flies straight)
- Vertical velocity (vz): Mostly zero (expert maintains altitude)

**Distribution estimate**:
- 80-90% of labels: `[vx, ~0, ~0]` (straight forward flight)
- 10-20% of labels: `[vx, vy≠0, vz≠0]` (obstacle avoidance)

### Degenerate Minimum

The model found a **minimum-loss shortcut**:
1. Always predict `[vx_in, 0, 0]` (fly straight forward)
2. This is correct 80-90% of the time in training data
3. Achieves low validation loss (0.009)
4. But fails at any obstacle requiring lateral/vertical avoidance

**Why it works in training**:
- MSE loss: `L = (vx_pred - vx_true)² + (vy_pred - vy_true)² + (vz_pred - vz_true)²`
- If `vy_true ≈ 0` and `vz_true ≈ 0` most of the time, predicting `vy=0, vz=0` minimizes loss
- Model learns to ignore image and just copy forward velocity

---

## Comparison: Epoch 1 vs Epoch 92

### Epoch 1 (Old, "Worse" Model)

**Metrics**:
- Validation loss: 0.019 (higher)
- Simulation: 20m goal reached, 0 collisions ✓

**Behavior**:
- Outputs vary meaningfully across inputs
- vx ranges: -2.4 to -1.9 m/s
- vy ranges: +0.19 to +0.75 m/s
- vz ranges: +0.02 to -0.79 m/s
- **Model responds to image content**

**Why it works**: Model is still learning, hasn't collapsed yet. Outputs are noisy but responsive.

### Epoch 92 (New, "Better" Model)

**Metrics**:
- Validation loss: 0.009 (lower)
- Simulation: Crashed at 3.98s, 1 collision ✗

**Behavior**:
- Outputs are nearly constant across inputs
- vx: ~4.0 m/s (copies input)
- vy: ~-0.007 m/s (constant)
- vz: ~+0.07 m/s (constant)
- **Model ignores image content**

**Why it fails**: Model collapsed to degenerate solution. Flies straight into first obstacle.

---

## Evidence

### 1. Deterministic Crash

- Always crashes at exactly 3.98s
- Always at same location (x≈20m, segment 20)
- Consistent across multiple runs

**Interpretation**: Model flies straight forward at full speed with no avoidance, hits first unavoidable obstacle.

### 2. Input Invariance

Tested with 20 random depth images:
- All outputs: `[4.069, -0.007, +0.07]` ± 0.02
- Output variance near zero
- No correlation with image content

**Interpretation**: Model doesn't use image information.

### 3. Velocity Echo Pattern

Model output scales linearly with input velocity:
- `vx_out ≈ vx_in * 1.0`
- `vy_out ≈ 0` (constant)
- `vz_out ≈ 0` (constant)

**Interpretation**: Model learned identity function for forward velocity, zero function for lateral/vertical.

---

## Why Inference Code Changes Didn't Help

We tried multiple fixes to `user_code.py`:
1. Removed aggressive velocity scaling (lines 148-164)
2. Removed startup ramp (lines 145-150)
3. Let model control velocity directly

**Result**: No improvement - still crashes at 3.98s

**Reason**: The problem is in the **model weights**, not the inference code. The model itself outputs `[~4.0, ~0, ~0]` regardless of input, so no amount of inference code changes can fix it.

---

## Solution: Retrain with Balanced Loss

### Problem

Standard MSE loss treats all dimensions equally:
```python
loss = (vx_pred - vx_true)² + (vy_pred - vy_true)² + (vz_pred - vz_true)²
```

With imbalanced labels (vy≈0, vz≈0 most of the time), the model learns to always predict vy=0, vz=0.

### Solution 1: Weighted Loss

Weight lateral/vertical errors more heavily:
```python
loss = (vx_pred - vx_true)² + 
       5.0 * (vy_pred - vy_true)² + 
       5.0 * (vz_pred - vz_true)²
```

This forces the model to pay attention to rare but critical lateral/vertical corrections.

### Solution 2: Variance Regularization

Add penalty for low output variance:
```python
loss = mse_loss + 
       lambda * (1.0 / (var(vy_pred) + epsilon)) + 
       lambda * (1.0 / (var(vz_pred) + epsilon))
```

This prevents the model from collapsing to constant outputs.

### Solution 3: Data Augmentation

Oversample obstacle avoidance scenarios:
- Collect more data with lateral/vertical maneuvers
- Augment existing data with synthetic obstacles
- Balance the label distribution

### Solution 4: Multi-Task Learning

Add auxiliary tasks that require image understanding:
- Obstacle detection (binary classification)
- Distance to nearest obstacle (regression)
- Collision prediction (binary classification)

This forces the model to use image information.

---

## Recommended Actions

### Immediate (Use Old Weights)

**Action**: Revert to epoch=1 weights for production deployment

**Rationale**:
- Epoch=1 works (20m goal, 0 collisions)
- Epoch=92 fails (crashes at 3.98s)
- Lower validation loss doesn't mean better performance

**Files**:
- Use: `experiments/mamba_branches/optimized_training/branch_B/best_model.pth` (epoch=1)
- Don't use: `training/checkpoints/branch_B_best.pth` (epoch=92)

### Short-Term (Test Intermediate Epochs)

**Action**: Test checkpoints at epochs 25, 50, 75

**Rationale**:
- Find the epoch before collapse occurred
- May find a sweet spot with good loss and good behavior

**Files**:
- `experiments/mamba_branches/optimized_training/branch_B/checkpoint_epoch_25.pth`
- `experiments/mamba_branches/optimized_training/branch_B/checkpoint_epoch_50.pth`
- `experiments/mamba_branches/optimized_training/branch_B/checkpoint_epoch_75.pth`

**Warning**: These checkpoints show near-zero loss (0.000003 to 0.0000006), suggesting they may have the same collapse problem. Test with input-sensitivity check before using.

### Long-Term (Retrain with Fixes)

**Action**: Retrain all branches with balanced loss and data augmentation

**Changes needed**:
1. Implement weighted loss (5x weight on vy/vz errors)
2. Add variance regularization to prevent collapse
3. Augment training data with more obstacle avoidance scenarios
4. Add auxiliary tasks (obstacle detection, collision prediction)
5. Monitor output variance during training (early warning of collapse)

**Expected outcome**:
- Models that respond to image content
- Non-zero lateral/vertical control
- Successful obstacle avoidance

---

## Lessons Learned

### 1. Lower Loss ≠ Better Performance

Validation loss is a proxy metric, not the true objective. A model can achieve low loss by learning a degenerate shortcut that fails in practice.

### 2. Imbalanced Labels Cause Collapse

When some labels are rare (vy≠0, vz≠0), the model learns to ignore them and always predict the common case (vy=0, vz=0).

### 3. Monitor Output Variance

If model outputs become constant during training, it's a sign of collapse. Add variance monitoring to training loop.

### 4. Test with Input Sensitivity

Don't just check validation loss - verify that model outputs actually vary with input. A model that ignores input can still have low loss.

### 5. Early Stopping Can Help

Epoch=1 worked better than epoch=92. Sometimes stopping early prevents collapse.

---

## Appendix: Diagnostic Commands

### Check Model Output Variance

```python
import torch
import numpy as np

model = load_model('path/to/checkpoint.pth')
images = [random_depth_image() for _ in range(20)]

outputs = [model(img) for img in images]
vy_values = [out[1] for out in outputs]
vz_values = [out[2] for out in outputs]

print(f"vy variance: {np.var(vy_values)}")
print(f"vz variance: {np.var(vz_values)}")

# If variance < 0.001, model has collapsed
```

### Check Input-Output Relationship

```python
# Test if model echoes input velocity
for vx_in in [1.0, 4.0, 6.0]:
    output = model(random_image, vx_in)
    print(f"vx_in={vx_in} → vx_out={output[0]}, ratio={output[0]/vx_in}")

# If ratio ≈ 1.0 for all inputs, model is copying input
```

### Check Image Sensitivity

```python
# Test if model responds to different images
img1 = load_image('obstacle_left.png')
img2 = load_image('obstacle_right.png')
img3 = load_image('clear_path.png')

out1 = model(img1)
out2 = model(img2)
out3 = model(img3)

print(f"Output variance: {np.var([out1, out2, out3], axis=0)}")

# If variance ≈ 0, model ignores image
```

---

**Report Generated**: 2026-05-02  
**Analysis Method**: Input-output sensitivity testing, variance analysis, crash pattern analysis  
**Recommendation**: Use epoch=1 weights, retrain with balanced loss
