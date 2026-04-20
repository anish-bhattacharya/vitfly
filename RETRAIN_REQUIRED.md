# Retrain Required — Training Label Bug

**Branch:** `mambatest` | **Commit:** `6d668b5` | **Date:** 2026-04-20

## TL;DR for the next agent

All 5 Mamba branches (A-E) were sim-tested and all crash immediately. Root cause is a **training label bug** in `training/train_mamba_optimized.py`. Fix the label, retrain, then run `./run_mamba_eval_all.bash 10`. The simulation infrastructure is already working.

---

## Root Cause

**File:** `training/train_mamba_optimized.py`, lines 81–91

```python
# CURRENT CODE — WRONG
velocity = torch.from_numpy(self.traj_meta[idx, 2:5]).float()  # desired vel input
target = velocity.clone()  # BUG: target == input → model learns identity mapping
```

The model is trained to output the same value it receives as input. It never learns to avoid obstacles — it just echoes the desired velocity direction.

**Correct target** (from ViTLSTM baseline `vitfly/training/train.py:170`):
```python
self.train_velcmd = self.train_meta[:, range(13, 16)]  # expert linvel command
```

---

## Required Fix

In `training/train_mamba_optimized.py`, change `__getitem__`:

```python
# BEFORE (wrong)
velocity = torch.from_numpy(self.traj_meta[idx, 2:5]).float()
target = velocity.clone()

# AFTER (correct)
velocity = torch.from_numpy(self.traj_meta[idx, 2:5]).float()   # input: desired vel (unchanged)
target = torch.from_numpy(self.traj_meta[idx, 13:16]).float()   # target: expert linvel cmd
target = target / (torch.norm(target) + 1e-6)                   # normalize by magnitude
```

> **Column note:** PNG-format data uses cols `13:16`. Non-PNG uses `12:15`.
> Check your dataset format — see `vitfly/training/train.py:170` for reference.

---

## Retrain Commands

```bash
cd /root/catkin_ws/src/vitfly-mambatest
conda activate ros_py38  # Python 3.8, PyTorch 2.4.1+cu118

# Fix the label bug first, then:
python3 training/train_mamba_optimized.py --branch B --model_type MambaVisionSSM
python3 training/train_mamba_optimized.py --branch C --model_type CNNMamba3 --ssm_d_state 16
python3 training/train_mamba_optimized.py --branch D --model_type STHMamba
python3 training/train_mamba_optimized.py --branch E --model_type DecisionMamba
python3 training/train_mamba_optimized.py --branch A --model_type VMambaLSTM
```

Weights go to `experiments/mamba_branches/optimized_training/branch_{A-E}/best_model.pth`.

---

## Validate After Retraining

```bash
# Set IP alias (required every WSL restart)
ip addr add 192.168.233.250/32 dev lo

source /root/catkin_ws/devel/setup.bash
cd /root/catkin_ws/src/vitfly-mambatest

# Run all 5 branches, 10 rollouts each
./run_mamba_eval_all.bash 10
```

Results saved to `results/mamba_eval_<timestamp>/branch_{A-E}/evaluation.yaml`.
Success criterion: `Success: true` in at least some rollouts (ViTLSTM baseline achieves ~60-70%).

---

## Simulation Infrastructure Status (no changes needed)

| File | Status | Notes |
|------|--------|-------|
| `envtest/ros/run_competition.py` | ✅ Ready | Loads all 5 branch models via `--model_type` arg |
| `envtest/ros/user_code.py` | ✅ Ready | Correct velocity shapes: VMambaLSTM=3D+LSTM, B/C/D/E=3D stateless |
| `launch_mamba_evaluation.bash` | ✅ Ready | Parameterized: `bash launch_mamba_evaluation.bash <N> vision "" <MODEL_TYPE> <MODEL_PATH>` |
| `run_mamba_eval_all.bash` | ✅ Ready | Loops A-E, checks IP alias, saves per-branch results |

---

## Model Details (for reference)

| Branch | Class | Velocity input | Hidden state |
|--------|-------|---------------|--------------|
| A | `VMambaLSTMNet` | `(B,3)` 3D | Yes — LSTM `(h,c)` |
| B | `MambaVisionSSMNet` | `(B,3)` 3D | No |
| C | `CNNMamba3Net` | `(B,3)` 3D | No — **must use `ssm_d_state=16`** |
| D | `STHMambaNet` | `(B,3)` 3D | No |
| E | `DecisionMambaNet` | `(B,3)` 3D | No |

Branch C checkpoint was trained with `ssm_d_state=16` (not the default 32). This is already handled in `run_competition.py`.

---

## Why `train_mamba_branch.py` is NOT the right script

`training/mamba_infra/train_mamba_branch.py` uses the correct `velcmd = traj_meta[:, 13:16]` target, but it was used for the earlier per-branch experiments (not the `optimized_training` weights). The `optimized_training` weights were all produced by `train_mamba_optimized.py` which has the bug. Fix `train_mamba_optimized.py` and retrain — don't switch to `train_mamba_branch.py` unless you want to restructure the experiment layout.
