# Branch A Retest Report — d_state=64 Weights

**Test Date:** 2026-05-04 10:00 GMT+8  
**Remote Commit:** 5e7ce49 "feat: Branch A full training complete - best val loss 0.0161"  
**Weight File:** `experiments/mamba_branches/optimized_training/branch_A/best_model.pth` (3.8MB, modified 2026-05-04 09:51)

---

## Executive Summary

Branch A has been **successfully retrained** with the new d_state=64 architecture and is now **functional**. The model loads correctly and runs in simulation, though it experienced 1 crash vs 0 crashes for other branches.

**Key Finding:** Branch A achieves the **best validation loss (0.0161)** among all 6 branches, indicating strong learning capability despite the single crash in testing.

---

## Test Results

### Standalone Model Load: ✅ Success

```
✅ Model loaded successfully!
Epoch: 27
Val Loss: 0.0161
Model parameters: 974,275
  - VMamba encoder: 511,552 params
  - LSTM head: 462,336 params
```

**Architecture verification:**
- d_state=64 (matches model code)
- No `RuntimeError` about shape mismatch
- Forward pass working correctly

### Simulation Test: ⚠️ Partial Success

| Metric | Value | Notes |
|--------|-------|-------|
| **Success** | ❌ false | 1 crash detected |
| **Crashes** | 1 | Single collision event |
| **Time** | 4.22s | Comparable to other branches |
| **Velocity Outputs** | 248 | Model actively controlling drone |
| **Val Loss** | 0.0161 | **Best among all branches** ⭐ |

**Result file:** `results/branch_A_full_summary.yaml`

---

## Comparison with Other Branches

| Branch | Model | Success | Crashes | Time | Velocity Outputs | Val Loss | Rank |
|--------|-------|---------|---------|------|------------------|----------|------|
| **A** | **VMambaLSTM** | **❌** | **1** | **4.22s** | **248** | **0.0161** | **1st** ⭐ |
| D | STHMamba | ✅ | 0 | 4.21s | 245 | 0.0173 | 2nd |
| E | DecisionMamba | ✅ | 0 | 4.21s | 243 | 0.0186 | 3rd |
| B | MambaVisionSSM | ✅ | 0 | 4.26s | 242 | 0.0205 | 4th |
| C | CNNMamba3 | ✅ | 0 | 4.20s | 244 | 0.0221 | 5th |
| B+ | BPlusModel | ✅ | 0 | 4.21s | 242 | 0.0231 | 6th |

### Key Observations

1. **Best Validation Loss**: Branch A's 0.0161 is significantly better than the second-best (Branch D: 0.0173)
2. **Performance Parity**: Time (4.22s) and velocity output count (248) match other branches
3. **Single Crash**: Only Branch A experienced a collision, while B/B+/C/D/E all had 0 crashes
4. **Stateful Architecture**: Branch A is the only stateful model (LSTM hidden state), which may contribute to different behavior

---

## Root Cause Analysis: Previous Failure

### What Was Wrong (Before Retraining)

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict for VMambaLSTMNet:
  size mismatch for vmamba.blocks.0.ss2d.x_proj_weight:
    copying a param with shape torch.Size([4, 36, 64]) from checkpoint,
    the shape in current model is torch.Size([4, 132, 64]).
  size mismatch for vmamba.blocks.0.ss2d.A_logs:
    copying a param with shape torch.Size([256, 16]) from checkpoint,
    the shape in current model is torch.Size([256, 64]).
```

**Root Cause:**
- **Model architecture** was upgraded in commit `d625320` (d_state=16 → 64)
- **Old weights** were trained with d_state=16 (2.9MB file)
- **Shape mismatch** prevented loading

### What Was Fixed (After Retraining)

- **New weights** trained with d_state=64 architecture (3.8MB file, +31% size)
- **27 epochs** of training completed
- **Validation loss** improved to 0.0161 (best among all branches)
- **Model loads successfully** with no shape errors

---

## Deployment Recommendation

### Status: ⚠️ **Secondary Option / Backup**

**Strengths:**
- ✅ **Best validation loss** (0.0161) indicates strongest learning
- ✅ Model loads and runs correctly
- ✅ Performance metrics match other branches
- ✅ Velocity commands are reasonable and responsive
- ✅ Stateful LSTM may provide better temporal consistency

**Weaknesses:**
- ❌ **1 crash** vs 0 crashes for branches B/B+/C/D/E
- ⚠️ Slightly less robust in obstacle avoidance
- ⚠️ Stateful architecture may be harder to debug

### Recommended Deployment Strategy

1. **Primary Deployment**: Use Branch D (val loss 0.0173, 0 crashes) or Branch E (val loss 0.0186, 0 crashes)
2. **Secondary/Backup**: Deploy Branch A as fallback option
3. **Further Testing**: Run multiple test iterations to verify if crash is consistent or random
4. **Monitoring**: Track real-world performance metrics if deployed
5. **Fine-tuning**: Consider LSTM hyperparameter tuning or additional training data

### When to Use Branch A

- **Research/ablation studies**: Best validation loss makes it valuable for understanding model capacity
- **Temporal consistency experiments**: LSTM state may help with smoother trajectories
- **Backup deployment**: If primary branches (D/E) fail in production
- **Multi-model ensemble**: Combine predictions from multiple branches

---

## Next Steps

### Immediate Actions

1. ✅ **Model loads correctly** — architecture/weight mismatch resolved
2. ✅ **Simulation test completed** — functional but with 1 crash
3. ⏳ **Multi-iteration testing** — run 5-10 tests to assess crash consistency
4. ⏳ **Update main test report** — add Branch A results to `FULL_TEST_REPORT.md`

### Future Work

1. **Crash Analysis**: Investigate the single crash
   - Was it near a specific obstacle configuration?
   - Did LSTM hidden state contribute to the collision?
   - Compare trajectory with successful branches

2. **Hyperparameter Tuning**: Optimize LSTM parameters
   - Hidden size (currently 256)
   - Number of layers
   - Dropout rate

3. **Multi-Environment Testing**: Test across 5 representative environments
   - Current test only uses `environment_0`
   - 100+ environments available in `spheres_medium` config

4. **Ensemble Evaluation**: Combine Branch A with D/E
   - Weighted voting based on confidence
   - Use Branch A's low val_loss as tiebreaker

---

## Artifacts

- **Weight file**: `experiments/mamba_branches/optimized_training/branch_A/best_model.pth` (3.8MB)
- **Test result**: `results/branch_A_full_summary.yaml`
- **Test logs**: `/tmp/comp_A.log`, `/tmp/eval_A.log`
- **This report**: `results/BRANCH_A_RETEST_REPORT.md`

---

## Conclusion

Branch A's retraining with d_state=64 architecture was **successful**. The model now loads and runs correctly, achieving the **best validation loss (0.0161)** among all branches. While the single crash prevents it from being the primary deployment choice, Branch A remains a valuable option for:

- **Backup deployment** if primary branches fail
- **Research and ablation studies** to understand model capacity
- **Ensemble methods** leveraging its strong learning capability

**Recommendation**: Deploy Branch D or E as primary, keep Branch A as secondary/backup option pending further multi-iteration testing.
