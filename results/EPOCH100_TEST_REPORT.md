# 100-Epoch Training Test Report
**Test Date:** 2026-05-02  
**Working Directory:** `/root/catkin_ws/src/vitfly-mambatest`  
**Git Commit:** 2104b43 (pulled 100-epoch trained weights)

## Executive Summary

**CRITICAL FINDING:** Full 100-epoch training resulted in **REGRESSION** for most branches.

- **Only 1 of 5 branches succeeded** (Branch B: MambaVisionSSM)
- **Branches D and E regressed** from successful (epoch-1) to failed (epoch-100)
- **Branch A remained failed** (no improvement)
- **Branch C remained failed** (no improvement)

**Recommendation:** ❌ **NOT READY FOR DEPLOYMENT**

## Detailed Results Comparison

### Branch A: VMambaLSTM
| Metric | Epoch-1 | Epoch-100 | Change |
|--------|---------|-----------|--------|
| **Success** | ❌ Failed | ❌ Failed | No change |
| **Crashes** | 1 | 1 | Same |
| **Time** | 4.45s | 4.21s | -0.24s |
| **Velocity Outputs** | 217 | 217 | Same |
| **vy Range** | ~1.86 (constant) | [0.02, 0.04] | ✅ Fixed mode collapse |
| **vy Variation** | None (constant) | Low variation | ✅ Improved |

**Analysis:** Training fixed the mode collapse issue (constant 1.86 output), but the model still crashes. Velocity outputs now show small positive values with low variation.

---

### Branch B: MambaVisionSSM ⭐
| Metric | Epoch-1 | Epoch-100 | Change |
|--------|---------|-----------|--------|
| **Success** | ❌ Failed | ✅ **SUCCESS** | ✅ **FIXED** |
| **Crashes** | 1 | 0 | ✅ Eliminated |
| **Time** | 4.36s | 4.22s | -0.14s |
| **Velocity Outputs** | 241 | 241 | Same |
| **vy Range** | [-0.87, +0.73] | [-0.17, -0.15] | More consistent |
| **vy Variation** | High variation | Good variation | Stable |

**Analysis:** ✅ **ONLY SUCCESSFUL BRANCH!** Training successfully fixed the crash issue. Velocity outputs show consistent negative vy values (-0.15 to -0.17), indicating stable left steering behavior. This is the **BEST PERFORMING MODEL**.

---

### Branch C: CNNMamba3
| Metric | Epoch-1 | Epoch-100 | Change |
|--------|---------|-----------|--------|
| **Success** | ❌ Failed | ❌ Failed | No change |
| **Crashes** | 1 | 1 | Same |
| **Time** | 4.23s | 4.21s | -0.02s |
| **Velocity Outputs** | 244 | 244 | Same |
| **vy Range** | [+0.20, +0.23] | [0.06, 0.35] | Wider range |
| **vy Variation** | Low variation | Good variation | ✅ Improved |

**Analysis:** Training improved velocity variation (wider range 0.06-0.35), but the model still crashes. Shows active right steering but fails to complete the course.

---

### Branch D: STHMamba ⚠️
| Metric | Epoch-1 | Epoch-100 | Change |
|--------|---------|-----------|--------|
| **Success** | ✅ Success | ❌ **FAILED** | ❌ **REGRESSION** |
| **Crashes** | 0 | 1 | ❌ New crash |
| **Time** | 4.20s | 4.20s | Same |
| **Velocity Outputs** | 245 | 245 | Same |
| **vy Range** | [+0.22, +0.26] | [0.18, 0.23] | Similar |
| **vy Variation** | Good variation | Good variation | Same |

**Analysis:** ❌ **CRITICAL REGRESSION!** This branch worked perfectly with epoch-1 weights but now crashes with epoch-100 weights. Velocity outputs remain similar, suggesting the issue is not mode collapse but rather overfitting or training instability.

---

### Branch E: DecisionMamba ⚠️
| Metric | Epoch-1 | Epoch-100 | Change |
|--------|---------|-----------|--------|
| **Success** | ✅ Success | ❌ **FAILED** | ❌ **REGRESSION** |
| **Crashes** | 0 | 1 | ❌ New crash |
| **Time** | 4.20s | 4.22s | +0.02s |
| **Velocity Outputs** | 247 | 247 | Same |
| **vy Range** | [+0.007, +0.009] | [0.0007, 0.0064] | Similar |
| **vy Variation** | Very low | Very low | Same |

**Analysis:** ❌ **CRITICAL REGRESSION!** This branch worked perfectly with epoch-1 weights but now crashes with epoch-100 weights. Velocity outputs remain near-zero (minimal steering), suggesting the model may have learned to be too conservative.

---

## Summary Statistics

### Success Rate Comparison
| Training Stage | Success Rate | Successful Branches |
|----------------|--------------|---------------------|
| **Epoch-1** | 40% (2/5) | D, E |
| **Epoch-100** | 20% (1/5) | B |
| **Change** | ❌ -20% | Net loss of 1 |

### Crash Analysis
| Branch | Epoch-1 Crashes | Epoch-100 Crashes | Change |
|--------|-----------------|-------------------|--------|
| A | 1 | 1 | Same |
| B | 1 | 0 | ✅ Fixed |
| C | 1 | 1 | Same |
| D | 0 | 1 | ❌ New crash |
| E | 0 | 1 | ❌ New crash |
| **Total** | 3 | 4 | ❌ +1 crash |

---

## Key Findings

### 1. Training Helped Only Branch B
- Branch B (MambaVisionSSM) is the **only success story**
- Fixed crash issue and achieved stable flight
- Consistent negative vy steering (-0.15 to -0.17)

### 2. Critical Regressions in Branches D & E
- Both branches **worked with epoch-1** but **failed with epoch-100**
- Suggests **overfitting** or **training instability**
- Velocity outputs remain similar, so not a mode collapse issue

### 3. Branches A & C Show No Improvement
- Branch A fixed mode collapse but still crashes
- Branch C improved velocity variation but still crashes
- Both need architectural or training changes

### 4. Velocity Output Analysis
- **Branch A:** Small positive vy (0.02-0.04) - low variation
- **Branch B:** Negative vy (-0.15 to -0.17) - stable steering ✅
- **Branch C:** Wide positive vy (0.06-0.35) - good variation
- **Branch D:** Moderate positive vy (0.18-0.23) - good variation
- **Branch E:** Near-zero vy (0.0007-0.0064) - minimal steering

---

## Root Cause Analysis

### Why Did Branches D & E Regress?

**Hypothesis 1: Overfitting**
- Models may have overfit to training data
- Lost generalization ability that epoch-1 weights had
- Need to check training/validation loss curves

**Hypothesis 2: Training Instability**
- Training may have diverged after early epochs
- Best model selection may have picked a suboptimal checkpoint
- Need to review training logs and loss curves

**Hypothesis 3: Learning Rate Issues**
- Learning rate may have been too high
- Caused model to "forget" good early behavior
- Need to review optimizer settings

---

## Recommendations

### Immediate Actions

1. **Deploy Branch B (MambaVisionSSM) for Testing**
   - Only successful model with 100-epoch weights
   - Shows stable flight and consistent steering
   - Ready for real-world validation

2. **Investigate Branches D & E Regression**
   - Review training logs and loss curves
   - Check for overfitting indicators
   - Consider using epoch-1 weights for deployment

3. **Analyze Training Dynamics**
   - Plot training/validation loss over epochs
   - Identify when Branches D & E started degrading
   - Determine optimal early stopping point

### Long-Term Actions

1. **Implement Early Stopping**
   - Monitor validation performance during training
   - Stop training when validation loss plateaus
   - Save best model based on validation metrics

2. **Add Regularization**
   - Increase dropout rates
   - Add weight decay
   - Use data augmentation

3. **Hyperparameter Tuning**
   - Reduce learning rate
   - Adjust batch size
   - Experiment with different optimizers

4. **Architecture Review**
   - Analyze why Branch B succeeded
   - Consider adopting MambaVisionSSM architecture for other branches
   - Investigate architectural differences

---

## Deployment Decision

### ❌ NOT READY FOR FULL DEPLOYMENT

**Reasons:**
1. Only 1 of 5 branches succeeded (20% success rate)
2. Two branches regressed from working to failing
3. Training process shows instability
4. Need to investigate root causes before scaling

### ✅ READY FOR LIMITED TESTING

**Branch B (MambaVisionSSM) can be deployed for:**
- Controlled testing environment
- Real-world validation
- Performance benchmarking
- Data collection for further training

**Conditions:**
- Monitor closely for crashes
- Collect flight telemetry
- Compare with baseline performance
- Be ready to revert to epoch-1 weights if issues arise

---

## Next Steps

1. **Immediate:** Deploy Branch B for controlled testing
2. **Short-term:** Investigate Branches D & E regression
3. **Medium-term:** Retrain with early stopping and regularization
4. **Long-term:** Adopt successful architecture patterns from Branch B

---

## Test Artifacts

- Branch A summary: `results/branch_A_100epoch_summary.yaml`
- Branch B summary: `results/branch_B_100epoch_summary.yaml`
- Branch C summary: `results/branch_C_100epoch_summary.yaml`
- Branch D summary: `results/branch_D_100epoch_summary.yaml`
- Branch E summary: `results/branch_E_100epoch_summary.yaml`
- Test logs: `/tmp/branch_*_100epoch_test.log`

---

**Report Generated:** 2026-05-02  
**Tested By:** Automated test suite (`test_mamba_branch.bash`)  
**Model Weights:** 100-epoch trained (commit 2104b43)
