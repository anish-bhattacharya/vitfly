# ViTLSTM Baseline Model Issue - mambatest-distill Branch

**Date:** 2026-05-04  
**Branch:** mambatest-distill  
**Status:** ❌ BLOCKED - Model loading failure

---

## Issue Summary

ViTLSTM baseline model fails to load due to input dimension mismatch between the trained checkpoint and current model architecture.

## Error Details

```
RuntimeError: Error(s) in loading state_dict for LSTMNetVIT:
  size mismatch for lstm.weight_ih_l0: 
    copying a param with shape torch.Size([512, 517]) from checkpoint, 
    the shape in current model is torch.Size([512, 519])
```

**Location:** `envtest/ros/run_competition.py` line 176

## Root Cause

- **Checkpoint**: Trained with 517-dimensional LSTM input
- **Current model**: Expects 519-dimensional LSTM input
- **Difference**: 2 extra features added to model architecture after training

## Impact

1. ❌ **Cannot run ViTLSTM baseline test** on mambatest-distill branch
2. ❌ **No depth images in RViz** - run_competition.py crashes before publishing depth_viz topic
3. ❌ **Drone stuck at takeoff height** - no velocity commands sent due to model crash
4. ❌ **Cannot generate summary.yaml** for baseline comparison

## Symptoms Observed

1. Simulator launches successfully
2. Unity connects and renders
3. Drone takes off to 3.5m height
4. run_competition.py crashes during model loading
5. evaluation_node keeps sending start_navigation commands (no response)
6. Drone remains stationary at (x=0, y=0, z=3.5)
7. RViz shows no depth_viz topic (because run_competition.py died)

## Model Information

- **Path:** `models/ViTLSTM_model.pth`
- **Size:** 14MB
- **Modified:** 2026-05-04 11:14
- **Architecture:** LSTMNetVIT
- **Expected input:** 517 dimensions (from checkpoint)
- **Current code:** 519 dimensions

## Likely Input Composition

**Original (517 dims):**
- ViT features: 512
- Velocity (vx, vy, vz): 3
- Unknown state: 2
- **Total:** 517

**Current (519 dims):**
- ViT features: 512
- Velocity (vx, vy, vz): 3
- Unknown state: 4 (2 extra features added)
- **Total:** 519

## Recommended Fix

**Option A (Preferred):** Adjust model code to match checkpoint
1. Find where the 2 extra features are concatenated
2. Remove or comment out the extra features
3. Restore 517-dimensional input
4. Test baseline runs successfully

**Option B (Not feasible now):** Retrain checkpoint with 519-dim architecture
- Requires full retraining cycle
- Not practical for immediate testing

## Files to Investigate

1. `envtest/ros/run_competition.py` - model loading and initialization
2. `envtest/ros/user_code.py` - model definition and forward pass
3. Model definition files in `models/` or `experiments/`
4. Look for input feature concatenation in forward() method

## Workaround

**For immediate testing:** Switch to `mambatest` branch where Mamba models work correctly.

```bash
git checkout mambatest
bash run_full_test.bash B MambaVisionSSM  # Test any Mamba branch
```

## Related Issues

- This is specific to the `mambatest-distill` branch
- The `mambatest` branch does not have this issue (uses Mamba models, not ViTLSTM)
- Suggests mambatest-distill modified the baseline architecture for distillation experiments

## Next Steps

1. ✅ **Document issue** (this file)
2. ⏳ **Switch to mambatest branch** for upstream simulation testing
3. ⏳ **Run Mamba branch tests** to verify upstream functionality
4. ⏳ **Fix mambatest-distill** when distillation work resumes
   - Identify the 2 extra features
   - Revert to 517-dimensional input
   - Or retrain ViTLSTM with 519-dim architecture

## Test Command (Failed)

```bash
cd /root/catkin_ws/src/vitfly-mambatest
git checkout mambatest-distill
bash launch_evaluation.bash 1 vision
```

**Result:** Model loading crash, no summary.yaml generated

---

**Conclusion:** The mambatest-distill branch has an incompatible ViTLSTM baseline model. Use the mambatest branch for functional testing until this is resolved.
