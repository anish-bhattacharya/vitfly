# Full Simulation Test Report — Latest Weights with summary.yaml

**Test Date:** 2026-05-03 21:30 GMT+8
**Working Directory:** `/root/catkin_ws/src/vitfly-mambatest`
**Git Commit:** 42be1a7 (latest mambatest)
**Test Method:** `run_full_test.bash` with persistent simulator + per-branch summary.yaml generation

## Executive Summary

5 of 6 branches passed full simulation testing. **Branch A failed to load** due to architecture/weight mismatch (intentional remote architecture change without corresponding retrain).

## Detailed Results

| Branch | Model Type | Result | Crashes | Time | Velocity Outputs | Notes |
|--------|-----------|--------|---------|------|------------------|-------|
| **A** | VMambaLSTM | ❌ **LOAD FAILED** | - | - | 0 | Architecture mismatch (see below) |
| B | MambaVisionSSM | ✅ Success | 0 | 4.26s | 242 | Stable behavior |
| B+ | BPlusModel | ✅ Success | 0 | 4.21s | 242 | New hybrid architecture works |
| C | CNNMamba3 | ✅ Success | 0 | 4.20s | 244 | Most velocity outputs |
| D | STHMamba | ✅ Success | 0 | 4.21s | 245 | Stable lateral steering |
| E | DecisionMamba | ✅ Success | 0 | 4.21s | 243 | Conservative steering |

**Files generated:** `results/branch_<X>_full_summary.yaml` for each branch.

## Branch A Critical Issue

### Symptom
`run_competition.py` raises `RuntimeError` during state_dict loading:
```
size mismatch for vmamba.blocks.0.ss2d.x_proj_weight:
  copying a param with shape torch.Size([4, 36, 64]) from checkpoint,
  the shape in current model is torch.Size([4, 132, 64]).
size mismatch for vmamba.blocks.0.ss2d.A_logs:
  copying a param with shape torch.Size([256, 16]) from checkpoint,
  the shape in current model is torch.Size([256, 64]).
```

### Root Cause
Commit `d625320` ("feat: mamba-ssm CUDA加速模块 + Branch A修复(d_state=64)") changed Branch A's `vmamba_encoder.py`:
- **Before**: `d_state=16`
- **After**: `d_state=64`

The model **architecture was intentionally upgraded** (per BRANCH_A_ANALYSIS.md — closer to the official VMamba paper specification). However, the weights at `experiments/mamba_branches/optimized_training/branch_A/best_model.pth` were trained with the **old `d_state=16` architecture** and cannot load into the new model.

### Required Action
**Retrain Branch A from scratch** with the new `d_state=64` architecture. The current weights are incompatible.

## RViz Observation: Post-Goal Collision

User observed in RViz: drones occasionally collide with obstacles **after** reaching the 20m goal. This is **not counted as a failure** because:
- `evaluation_node.py` stops monitoring after the 20m segment is reached
- Drones continue forward by inertia until colliding
- Summary correctly reports `Success: true, number_crashes: 0`

This is expected behavior of the evaluation framework. Real performance is measured **up to 20m only**.

## Recommendations for Remote Experiment Report

The remote training pipeline is the appropriate place to author a full experiment report. Suggestions for what to include:

### 1. Branch A — Retraining Required (Highest Priority)
- Architecture upgrade context: `d_state=16 → 64`, parallel scan integration (commit `c6d1c8a`)
- Why old weights can't be deployed: shape mismatch evidence
- Expected timeline: how long does Branch A retraining take?
- Backup recommendation: keep old `d_state=16` weights as `branch_A_d_state16_legacy.pth` for ablation comparison

### 2. Validation Loss vs Real Performance Disconnect
- Branch B epoch 1 (val_loss 0.0274) → Success ✅
- Branch B epoch 4 (val_loss 0.0205) → Success but with extreme `vz=2.98` near obstacles
- **Lower val_loss ≠ better real-world behavior**
- Suggest: training pipeline should track per-dimension output distribution (not just MSE)
  - If `max(|vz|)` exceeds threshold (e.g. 1.5) on validation set → flag as "panic-response" model

### 3. Output Variance Monitoring
- Mode collapse history: previous training runs produced near-constant outputs
- Current epoch-1 training escapes this, but no explicit guard
- Suggest: log `std(vy_pred)` and `std(vz_pred)` per epoch as standard metric

### 4. Per-Branch Architecture Notes
The repository now has 6 distinct architectures (A, B, B+, C, D, E). Each has different state-handling:
- **Stateful**: A (LSTM hidden state)
- **Stateless**: B, B+, C, D, E
- Branch A's stateful nature may be why it shows different failure modes

### 5. Multi-Environment Testing (Future Work)
- All current tests run on `environment_0` of `spheres_medium` config
- 100+ environments available in `flightmare/flightpy/configs/vision/spheres_medium/`
- Suggest: pick 5 representative environments (varying obstacle density) and report success rate as `X/5` per branch

### 6. Test Protocol Documentation
- Document the `run_full_test.bash` workflow (single-simulator-instance, sequential branch testing)
- Documents handle: Unity/RViz lifecycle, drone reset between branches, summary.yaml generation
- Reference vitfly skill: `.claude/skills/vitfly/SKILL.md`

## Test Infrastructure Notes

### Files Added/Modified This Session
- `run_full_test.bash` — Single-branch test script that reuses an already-running simulator (faster than `launch_mamba_evaluation.bash`)
- `launch_mamba_evaluation.bash` — Fixed IP changed from dynamic `hostname -I` to fixed `192.168.233.250` (matches loopback alias)
- `envtest/ros/run_competition.py` — Added BPlusModel import and loading
- `envtest/ros/user_code.py` — Added BPlusModel to stateless branch list
- `results/branch_<X>_full_summary.yaml` — Per-branch evaluation results

### Key Insight: Why Tests Sometimes "Fail"
RViz only displays depth images via the `/kingfisher/dodgeros_pilot/unity/depth_viz` topic, which is **published by `run_competition.py`**, not by the simulator. So:
- Empty RViz between test runs is **normal** (no inference node running)
- "No depth images" appears when manually starting simulator without inference node
- This was misdiagnosed earlier as a simulator failure

## Artifacts

- Per-branch summaries: `results/branch_<X>_full_summary.yaml`
- This report: `results/FULL_TEST_REPORT.md`
- Test logs: `/tmp/comp_<X>.log`, `/tmp/eval_<X>.log`
- Test infrastructure: `run_full_test.bash`

---

**Tested by:** vitfly skill + autoresearch debugging skill
**Weights commit:** 42be1a7
**Test commit:** [next commit hash]
