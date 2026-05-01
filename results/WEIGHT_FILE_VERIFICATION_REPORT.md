# Weight File Verification Report

**Date**: 2026-05-01  
**Verification Type**: Model weight file integrity check  
**Purpose**: Verify that Branch C/D/E weight files were correctly downloaded from GitHub

---

## Executive Summary

All weight files are **legitimate downloads from GitHub** (commit 0a96a2e, April 1, 2026). However, **Branch C has a critical issue**: the weight file was saved at epoch 1 with high validation loss (0.096), indicating incomplete training. This explains Branch C's poor simulation performance.

### Critical Finding

**Branch C (CNNMamba3)** weight file is **incomplete and unusable**:
- ❌ Only trained for **1 epoch** (validation loss: 0.096)
- ❌ Commit message claims "Best Val: 0.000001" but actual is 0.096
- ❌ Weight file saved too early and never updated
- ❌ Explains the 5 collisions and high lateral drift in simulation

**Branches D and E** weight files are **complete but trained on incorrect data**:
- ✅ Trained for 97-98 epochs (validation loss: ~10^-6 to 10^-7)
- ✅ Training converged properly
- ❌ But trained on data with velocity labels in wrong coordinate frame

---

## Detailed Verification Results

### Branch C (CNNMamba3)

**File Information**:
- Path: `experiments/mamba_branches/optimized_training/branch_C/best_model.pth`
- Size: 8.3 MB (8,605,579 bytes)
- Last modified: Apr 30 16:32 (today)
- Git status: Tracked, committed
- Git history: Added in commit 0a96a2e (Apr 1, 2026)
- MD5 hash: `f54bafbf0e8d9010da69ddc43e4809e5`

**Checkpoint Structure**:
```python
{
    'epoch': 1,                    # ❌ ONLY 1 EPOCH!
    'model_state_dict': {...},
    'val_loss': 0.096,             # ❌ HIGH VALIDATION LOSS
    'config': {...}
}
```

**Training Metadata**:
- Epoch: **1** (expected: 100)
- Validation loss: **0.096** (expected: <0.001)
- Has `_orig_mod.` prefix: No
- First layer: `cnn.stem.0.weight`
- Total parameters: 122

**Verdict**: ❌ **INCOMPLETE TRAINING - UNUSABLE**

**Root Cause**: Weight file was saved at the very beginning of training (epoch 1) and never updated. The model has barely learned anything, which explains:
- 5 collisions in simulation (highest among all branches)
- Excessive lateral drift (1.16-1.26 m/s)
- Early termination at ~4m
- Unstable navigation

**Action Required**: **Complete retraining from scratch** (100 epochs with corrected data)

---

### Branch D (STHMamba)

**File Information**:
- Path: `experiments/mamba_branches/optimized_training/branch_D/best_model.pth`
- Size: 11 MB (11,062,197 bytes)
- Last modified: Apr 30 16:32 (today)
- Git status: Tracked, committed
- Git history: Added in commit 0a96a2e (Apr 1, 2026)
- MD5 hash: `63b37a29f2a3b4f2ebb46e91d92439d3`

**Checkpoint Structure**:
```python
{
    'epoch': 97,                   # ✅ COMPLETE TRAINING
    'model_state_dict': {...},
    'val_loss': 9.1e-07,          # ✅ VERY LOW VALIDATION LOSS
    'config': {...}
}
```

**Training Metadata**:
- Epoch: **97** (expected: ~100) ✅
- Validation loss: **9.1×10^-7** (excellent) ✅
- Has `_orig_mod.` prefix: No
- First layer: `spatial_encoder.conv_stem.0.weight`
- Total parameters: 81

**Verdict**: ✅ **TRAINING COMPLETE** but ❌ **trained on incorrect data labels**

**Root Cause**: Training converged properly, but the training data had velocity labels in drone body frame instead of world frame. The model learned the wrong velocity-to-action mapping.

**Action Required**: **Retrain with corrected data labels** (world frame velocities)

---

### Branch E (DecisionMamba)

**File Information**:
- Path: `experiments/mamba_branches/optimized_training/branch_E/best_model.pth`
- Size: 5.2 MB (5,433,269 bytes)
- Last modified: Apr 30 16:32 (today)
- Git status: Tracked, committed
- Git history: Added in commit 0a96a2e (Apr 1, 2026)
- MD5 hash: `9ad766d509014d43d14452347be6398f`

**Checkpoint Structure**:
```python
{
    'epoch': 98,                   # ✅ COMPLETE TRAINING
    'model_state_dict': {...},
    'val_loss': 6.9e-06,          # ✅ VERY LOW VALIDATION LOSS
    'config': {...}
}
```

**Training Metadata**:
- Epoch: **98** (expected: ~100) ✅
- Validation loss: **6.9×10^-6** (excellent) ✅
- Has `_orig_mod.` prefix: No
- First layer: `patch_embed.proj.weight`
- Total parameters: 34

**Verdict**: ✅ **TRAINING COMPLETE** but ❌ **trained on incorrect data labels**

**Root Cause**: Same as Branch D - training converged properly, but trained on data with velocity labels in wrong coordinate frame.

**Action Required**: **Retrain with corrected data labels** (world frame velocities)

---

## Comparison Summary

| Branch | File Size | Epochs | Val Loss | Training Status | Data Labels | Verdict |
|--------|-----------|--------|----------|----------------|-------------|---------|
| **B** | 12MB | N/A | N/A | ✅ Complete | ✅ Correct | **PRODUCTION READY** |
| **C** | 8.3MB | **1** | **0.096** | ❌ Incomplete | ❌ Wrong frame | **UNUSABLE** |
| **D** | 11MB | 97 | 9.1×10^-7 | ✅ Complete | ❌ Wrong frame | **NEEDS RETRAINING** |
| **E** | 5.2MB | 98 | 6.9×10^-6 | ✅ Complete | ❌ Wrong frame | **NEEDS RETRAINING** |

---

## Simulation Performance Correlation

The weight file verification findings directly correlate with simulation test results:

### Branch C (1 epoch, val_loss=0.096)
- Collisions: **5** (worst)
- Lateral velocity: **1.16-1.26 m/s** (worst)
- Distance: **~4m** (failed to reach 20m goal)
- **Explanation**: Model barely trained, has no learned navigation strategy

### Branch D (97 epochs, val_loss=9.1×10^-7)
- Collisions: **3** (moderate)
- Lateral velocity: **0.97 m/s** (high)
- Distance: **~4m** (failed to reach 20m goal)
- **Explanation**: Model trained well but learned wrong velocity mapping

### Branch E (98 epochs, val_loss=6.9×10^-6)
- Collisions: **3** (moderate)
- Lateral velocity: **0.98 m/s** (high)
- Distance: **~4m** (failed to reach 20m goal)
- **Explanation**: Model trained well but learned wrong velocity mapping

---

## Verification Methodology

### 1. File Integrity Checks
```bash
# Check file sizes and timestamps
ls -lh experiments/mamba_branches/optimized_training/branch_*/best_model.pth

# Check git history
git log --follow --oneline experiments/mamba_branches/optimized_training/branch_C/best_model.pth

# Check git status
git status experiments/mamba_branches/optimized_training/branch_*/best_model.pth
```

### 2. Checkpoint Structure Analysis
```python
import torch

# Load checkpoints
ckpt_c = torch.load('branch_C/best_model.pth', map_location='cpu', weights_only=False)
ckpt_d = torch.load('branch_D/best_model.pth', map_location='cpu', weights_only=False)
ckpt_e = torch.load('branch_E/best_model.pth', map_location='cpu', weights_only=False)

# Inspect structure
print("Branch C:", ckpt_c.keys())
print("Epoch:", ckpt_c['epoch'])
print("Val loss:", ckpt_c['val_loss'])
```

### 3. Hash Verification
```bash
# Verify files are unique (not copies)
md5sum experiments/mamba_branches/optimized_training/branch_*/best_model.pth
```

All three files have unique MD5 hashes, confirming they are different models.

### 4. Prefix Check
```python
# Check for torch.compile() prefix
state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) else ckpt
has_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
```

None of the weight files have `_orig_mod.` prefix, indicating they were not trained with `torch.compile()`.

---

## Updated Retraining Plan

Based on these findings, the retraining plan needs to be adjusted:

### Priority 1: Branch C - Complete Retraining Required

**Status**: Current weights are **unusable** (only 1 epoch trained)

**Action**:
1. Correct training data labels (drone frame → world frame)
2. Train from scratch for **100 epochs**
3. Monitor validation loss (target: <0.001)
4. Save checkpoints every 25 epochs
5. Verify `best_model.pth` is actually the best checkpoint

**Expected Duration**: 2-4 hours (GPU dependent)

**Success Criteria**:
- Epoch ≥ 95
- Validation loss < 0.001
- Simulation test: 0 collisions, lateral velocity < 0.5 m/s

### Priority 2: Branches D and E - Retrain with Corrected Data

**Status**: Training complete but learned wrong mappings

**Action**:
1. Correct training data labels (drone frame → world frame)
2. Retrain for **100 epochs** with corrected data
3. Verify validation loss converges to <0.001
4. Test in simulation

**Expected Duration**: 2-4 hours per branch

**Success Criteria**:
- Validation loss < 0.001
- Simulation test: 0 collisions, lateral velocity < 0.5 m/s
- Successfully reach 20m goal

---

## Recommendations

### Immediate Actions

1. **Deploy Branch B to production** - only fully verified branch
2. **Flag Branch C weights as invalid** - add warning in README
3. **Correct training data labels** - convert all velocity labels to world frame
4. **Retrain all three branches** - C from scratch, D/E with corrected data

### Process Improvements

1. **Add training validation**:
   - Check epoch count before committing weights
   - Verify validation loss matches commit message
   - Add automated tests for weight file integrity

2. **Improve checkpoint saving**:
   - Save checkpoints every 25 epochs
   - Keep best checkpoint based on validation loss
   - Add metadata: training duration, hardware, data version

3. **Add simulation testing to CI/CD**:
   - Run quick simulation test before accepting weights
   - Verify forward velocity > 4.5 m/s
   - Verify collisions = 0

---

## Conclusion

The weight file verification revealed a critical issue with Branch C (incomplete training at epoch 1) and confirmed that Branches D and E have complete training but were trained on incorrectly labeled data.

**Key Findings**:
- ✅ All files are legitimate GitHub downloads (not local artifacts)
- ❌ Branch C: Training incomplete (1 epoch, val_loss=0.096)
- ✅ Branch D: Training complete (97 epochs, val_loss=9.1×10^-7)
- ✅ Branch E: Training complete (98 epochs, val_loss=6.9×10^-6)
- ❌ All three branches trained on data with velocity labels in wrong coordinate frame

**Action Required**:
1. Branch C: Complete retraining from scratch (100 epochs)
2. Branches D/E: Retrain with corrected data labels
3. All branches: Verify in simulation before deployment

**Timeline**: 6-12 hours total (2-4 hours per branch)

---

**Report Generated**: 2026-05-01  
**Verification Method**: Git history + PyTorch checkpoint inspection + MD5 hashing  
**Related Documents**:
- `MAMBA_BRANCH_TEST_REPORT.md` - Simulation test results
- `RETRAINING_PLAN.md` - Complete retraining procedure
