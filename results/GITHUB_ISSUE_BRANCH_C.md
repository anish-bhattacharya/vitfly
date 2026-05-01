# GitHub Issue: Branch C Weight File Incomplete - Requires Complete Retraining

## Issue Type
🐛 Bug / ⚠️ Critical Issue

## Priority
🔴 **HIGH** - Blocks production deployment of Branch C

## Summary

Branch C (CNNMamba3) model weight file is **incomplete and unusable** - saved at epoch 1 with validation loss of 0.096 instead of the claimed "Best Val: 0.000001". This explains the poor simulation performance (5 collisions, excessive lateral drift).

## Problem Description

### Weight File Status

**Branch C (CNNMamba3)**:
- ❌ Only trained for **1 epoch** (expected: 100)
- ❌ Validation loss: **0.096** (expected: <0.001)
- ❌ Commit message claims "Best Val: 0.000001" but actual checkpoint shows 0.096
- ❌ Weight file saved too early during training and never updated

**Verification Evidence**:
```python
# Checkpoint inspection
ckpt = torch.load('branch_C/best_model.pth')
print(ckpt['epoch'])      # Output: 1 (should be ~100)
print(ckpt['val_loss'])   # Output: 0.096 (should be <0.001)
```

**File Details**:
- Path: `experiments/mamba_branches/optimized_training/branch_C/best_model.pth`
- Size: 8.3 MB
- Git commit: 0a96a2e (Apr 1, 2026)
- MD5: `f54bafbf0e8d9010da69ddc43e4809e5`

### Impact on Simulation Performance

Branch C's incomplete training directly correlates with simulation failures:

| Metric | Branch C (1 epoch) | Branch B (complete) | Expected |
|--------|-------------------|---------------------|----------|
| Collisions | **5** | 0 | 0 |
| Lateral velocity | **1.16-1.26 m/s** | <0.2 m/s | <0.5 m/s |
| Distance covered | **~4m** | 20m | 20m |
| Goal completion | ❌ Failed | ✅ Success | ✅ Success |

**Root Cause**: Model has barely learned any navigation strategy due to only 1 epoch of training.

## Comparison with Other Branches

| Branch | Epochs | Val Loss | Training Status | Simulation Status |
|--------|--------|----------|----------------|-------------------|
| B (MambaVisionSSM) | N/A | N/A | ✅ Complete | ✅ Production ready |
| **C (CNNMamba3)** | **1** | **0.096** | ❌ **Incomplete** | ❌ **Unusable** |
| D (STHMamba) | 97 | 9.1×10^-7 | ✅ Complete | ⚠️ Needs data correction |
| E (DecisionMamba) | 98 | 6.9×10^-6 | ✅ Complete | ⚠️ Needs data correction |

**Note**: Branches D and E have complete training but were trained on data with velocity labels in wrong coordinate frame (separate issue).

## Required Actions

### 1. Complete Retraining (CRITICAL)

Branch C must be **completely retrained from scratch**:

**Prerequisites**:
- ✅ Correct training data labels (convert velocity from drone frame → world frame)
- ✅ Verify training script saves checkpoints correctly
- ✅ Monitor validation loss during training

**Training Configuration**:
```python
# Branch C (CNNMamba3) config
config = {
    'ssm_d_state': 16,
    'ssm_hidden': 256,
    'ssm_layers': 2,
    'dropout': 0.1
}

# Training parameters
epochs = 100
batch_size = 32
learning_rate = 1e-4
```

**Expected Duration**: 2-4 hours (GPU dependent)

**Success Criteria**:
- ✅ Epoch ≥ 95
- ✅ Validation loss < 0.001
- ✅ Checkpoint metadata matches actual training progress
- ✅ Simulation test: 0 collisions, lateral velocity < 0.5 m/s, reach 20m goal

### 2. Fix Checkpoint Saving Logic

Investigate why `best_model.pth` was saved at epoch 1 and never updated:

**Potential Issues**:
- Training script may have crashed after epoch 1
- `best_model.pth` saving logic may be broken
- Validation loss comparison may be inverted (saving worst instead of best)

**Verification**:
```python
# Check training script checkpoint saving logic
# Ensure it saves when val_loss improves, not at every epoch
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, 'best_model.pth')
```

### 3. Add Training Validation

Prevent this issue from happening again:

**Pre-commit Checks**:
```bash
# Verify checkpoint before committing
python3 << EOF
import torch
ckpt = torch.load('best_model.pth')
assert ckpt['epoch'] >= 95, f"Training incomplete: only {ckpt['epoch']} epochs"
assert ckpt['val_loss'] < 0.01, f"Val loss too high: {ckpt['val_loss']}"
print(f"✓ Checkpoint valid: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']}")
EOF
```

**CI/CD Integration**:
- Add automated checkpoint validation before accepting PRs
- Run quick simulation test to verify model produces reasonable outputs
- Compare checkpoint metadata with commit message claims

## Related Issues

- Training data labels in wrong coordinate frame (affects Branches D and E)
- See `WEIGHT_FILE_VERIFICATION_REPORT.md` for complete analysis
- See `RETRAINING_PLAN.md` for detailed retraining procedure

## Files to Update

After retraining:
- `experiments/mamba_branches/optimized_training/branch_C/best_model.pth` - Replace with properly trained weights
- `experiments/mamba_branches/optimized_training/branch_C/train_losses.npy` - Update training history
- `experiments/mamba_branches/optimized_training/branch_C/val_losses.npy` - Update validation history
- `results/MAMBA_BRANCH_TEST_REPORT.md` - Update with new test results
- `results/RETRAINED_COMPARISON.md` - Add performance comparison

## Testing Checklist

After retraining, verify:
- [ ] Checkpoint shows epoch ≥ 95
- [ ] Validation loss < 0.001
- [ ] Model loads without errors
- [ ] Forward pass produces 3D velocity output
- [ ] Simulation test: forward velocity ≥ 4.5 m/s
- [ ] Simulation test: lateral velocity < 0.5 m/s
- [ ] Simulation test: zero collisions
- [ ] Simulation test: successfully reach 20m goal
- [ ] `summary.yaml` generated with success=true

## Timeline

- **Data correction**: 2 hours (shared with Branches D/E)
- **Branch C retraining**: 2-4 hours
- **Validation testing**: 30 minutes
- **Total**: 4.5-6.5 hours

## References

- Commit 0a96a2e: Added incomplete Branch C weights (Apr 1, 2026)
- Commit d8acab5: Weight file verification report (May 1, 2026)
- Commit 8718ac2: Mamba branch test report and retraining plan (Apr 30, 2026)
- Commit 3aef004: RViz and velocity fixes (Apr 30, 2026)

## Labels

- `bug` - Incorrect weight file
- `priority: high` - Blocks production deployment
- `training` - Model training issue
- `branch-C` - Specific to CNNMamba3 branch
- `needs-retraining` - Requires complete retraining

---

**Created**: 2026-05-01  
**Status**: Open  
**Assignee**: TBD  
**Milestone**: Mamba Branch Retraining
