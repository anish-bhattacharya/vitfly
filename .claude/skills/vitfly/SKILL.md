---
name: vitfly
description: "Operational runbook for ROS1 Noetic + WSL2 + Flightmare simulation testing of Mamba drone obstacle avoidance models. Use when launching simulations, diagnosing failures, or running branch evaluations."
origin: local
---

# Vitfly Simulation Runbook

## When to Use

- Starting a simulation test session after WSL2 restart
- Diagnosing ZMQ port conflicts, IP mismatches, or ROS master failures
- Running evaluation tests for Mamba branches (A, B, B+, C, D, E)
- Diagnosing model load failures, mode collapse, or unexpected flight behavior
- Generating per-branch `summary.yaml` for reporting

---

## Experimental Design Matrix

Before running ANY simulation, identify where your test fits in the overall experimental matrix. This determines what conclusions the results support.

### Dimensions

| Dimension | Values | Status | Notes |
|-----------|--------|--------|-------|
| **Model** | Teacher (ViT+LSTM), A, B, B+, C, D, E | All 7 evaluated ✅ | Teacher = upstream best (7m/s real flight) |
| **Training** | BC baseline, Distill (α=β=γ=1.0) | BC + Distill ✅ | No ablation variants tested yet |
| **Track length** | 60m (upstream), 20m (early tests) | 60m is correct ✅ | 20m data was misleading — obstacles only fully sampled at 60m |
| **Desired velocity** | 5m/s (standard), 7m/s (teacher speed) | 5m/s @ 60m ✅; 7m/s @ 20m only ⚠️ | Teacher flies 7m/s but was only tested at 5m/s on 60m |
| **Prediction mode** | seq_len=1,2,4,8,16 | `run_competition.py` supports `--seq-len N` ✅ | All models only trained/evaluated at seq_len=1 so far. seq16 BC checkpoints available but overfit. **Need training pipeline to produce seq4/8/16 BC + distill checkpoints.** |

### Sequence Length Coverage

| Model | seq_len=1 | seq_len=4 | seq_len=8 | seq_len=16 |
|-------|-----------|-----------|-----------|------------|
| Teacher | ✅ 2/5 crashes | ⏳ | ⏳ | ⏳ |
| E BC | ✅ 3 crashes | ⏳ | ⏳ | ✅ 4 crashes (overfit) |
| E Distill | ✅ 1 crash 🏆 | ⏳ | ⏳ | ⏳ |
| B+ Distill | ✅ 1 crash 🏆 | ⏳ | ⏳ | ⏳ |
| Others | ✅ tested | ⏳ | ⏳ | ⏳ |

seq_len > 1 means the model receives N consecutive depth frames per inference. Stateful models (A, Teacher) maintain LSTM state across frames; stateless models (B/B+/C/D/E) use temporal attention/SSM across the sequence. **Seq_len=16 failed for BC because the checkpoint was overfit (epoch 100, val_loss 0.23), not because seq16 is inherently bad.** Proper seq16 distillation (`run_seq16_distill.sh`) may yield different results.

### Current Coverage (60m track, seq_len=1)

| Model | BC @ 5m/s | Distill @ 5m/s | Teacher @ 5m/s |
|-------|-----------|----------------|----------------|
| Teacher | — | — | ✅ 2 crashes |
| A | ✅ 3 crashes | ✅ 3 crashes | — |
| B | ❌ DNF | ✅ 2 crashes | — |
| B+ | ✅ 3 crashes | ✅ **1 crash** 🏆 | — |
| C | ✅ 3 crashes | ✅ 3 crashes | — |
| D | ✅ 2 crashes | ✅ 2 crashes | — |
| E | ✅ 3 crashes | ✅ **1 crash** 🏆 | — |

### What Has NOT Been Tested (Needs Training Pipeline)

- **seq_len=4 or 8 BC models** for any branch — need training pipeline to produce checkpoints
- **seq_len=4 or 8 distill models** — need training pipeline to produce checkpoints
- **Any distill model at seq_len > 1 in simulation** — inference pipeline now supports `--seq-len N`, but no multi-step checkpoints exist
- **seq16 distill** — `run_seq16_distill.sh` exists but hasn't produced a checkpoint yet
- **Loss weight ablation** (different α, β, γ) — all distill tests used α=β=γ=1.0
- **Teacher @ 7m/s on 60m track** — only tested at 5m/s (teacher native speed is 7m/s)
- **B+/E Distill @ 7m/s @ 60m** — only E tested at 7m/s @ 60m; B+ untested
- **init_from_bc distillation** — current distill is random init only

Always check this matrix before running a new test. If you're filling a gap, note it. If you're duplicating an existing result, skip it.

---

## Results Management

Every simulation run produces a `summary.yaml` with collision count and segment times. **Always save results with unique, descriptive filenames.** Never overwrite:

```bash
# ❌ DON'T — overwrites previous results
cp summary.yaml results/branch_D_distill_summary.yaml

# ✅ DO — include run identifier
cp summary.yaml results/branch_D_distill_60m_5ms_$(date +%m%d_%H%M).yaml
# Or use a descriptive tag + counter
cp summary.yaml results/branch_D_distill_60m_5ms_run2.yaml
```

The `run_full_test.bash` script saves to `results/branch_X_{variant}_summary.yaml` by default — this is fine for quick comparisons but will be overwritten on re-runs. For repeat experiments, use custom filenames.

Keep a results log in `results/RUN_LOG.md`:

```markdown
| Date | Branch | Variant | Speed | Track | Crashes | Time | Notes |
|------|--------|---------|-------|-------|---------|------|-------|
| 0505 | E | distill | 5m/s | 60m | 1 | 12.23s | ViT+LSTM teacher |
| 0505 | Teacher | — | 7m/s | 60m | 5 | 8.94s | native speed, worse |
```

---

## Environment Setup (WSL2 Network)

Run once per WSL2 session before any ROS commands:

```bash
# Add fixed loopback alias (required — do NOT use dynamic hostname -I)
ip addr add 192.168.233.250/32 dev lo 2>/dev/null

# Fix 127.0.0.1 routing (WSL2 mirrored mode breaks Unity-ZMQ otherwise)
if ip route get 127.0.0.1 2>/dev/null | grep -q loopback0; then
  ip route del 127.0.0.1 via 169.254.73.152 dev loopback0 proto kernel src 127.0.0.1 onlink table 127 2>/dev/null
  ip route flush cache 2>/dev/null
fi

# Verify: should show "dev lo", NOT "dev loopback0"
ip route get 127.0.0.1
```

Required environment variables (must export in EVERY shell that runs ROS commands — they don't persist between bash invocations):

```bash
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
export FLIGHTMARE_PATH=/root/catkin_ws/src/vitfly/flightmare
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
```

**Critical**: Always use the fixed IP `192.168.233.250`, never `$(hostname -I)`. The dynamic IP causes ROS master connection failures when the loopback alias is set to the fixed IP. The default shell has `ROS_MASTER_URI=http://localhost:11311` which **will not reach** the fixed-IP master — every manual `rostopic`/`roslaunch` invocation must export these variables first.

---

## Launch Sequence

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash --extend   # ← --extend preserves existing ROS_PACKAGE_PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
```

Remove stale roscore PID file (if roscore won't start after a crash):

```bash
rm -f /root/.ros/roscore-11311.pid
```

Launch simulator (roscore + Unity renderer + RViz in one command):

```bash
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
sleep 15  # wait for Unity ZMQ connection
```

**Verify topics are publishing** (after Unity connects):

```bash
rostopic list | grep -E "depth|state" | head -10
# Expected:
#   /kingfisher/dodgeros_pilot/unity/depth
#   /kingfisher/dodgeros_pilot/state
```

Reset and arm drone:

```bash
rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once
```

---

## Understanding RViz Depth Image Display (IMPORTANT)

**Common misconception**: "RViz isn't showing depth images, so the simulator is broken."

There are TWO separate depth topics:

| Topic | Publisher | When Active |
|-------|-----------|-------------|
| `/kingfisher/dodgeros_pilot/unity/depth` | Simulator (`visionsim_node`) | After simulator launches |
| `/kingfisher/dodgeros_pilot/unity/depth_viz` | `run_competition.py` (inference node) | **Only when inference is running** |

**RViz subscribes to `depth_viz`** (per `envsim/resources/rviz/envsim.rviz`). So:
- Simulator alone → RViz empty (this is **normal**, not a bug)
- Simulator + `run_competition.py` → RViz shows depth images
- Verify raw depth is publishing: `rostopic echo /kingfisher/dodgeros_pilot/unity/depth --noarr -n 1`

If raw `depth` topic exists but `depth_viz` doesn't, the simulator is fine — the inference node just isn't running yet.

---

## Post-Goal Collisions Are Not Counted

**Observation**: In RViz you may see the drone collide with an obstacle right after reaching 20m.

**Explanation**: `evaluation_node.py` stops monitoring after the 20m segment. The drone continues by inertia and may collide afterward, but this is **not** counted in `number_crashes`. A `Success: true, number_crashes: 0` result is valid even if RViz shows a post-goal collision.

---

## Model Testing Workflow

### Variant Testing (BC vs Distill)

Both `run_full_test.bash` and `test_mamba_branch.bash` support two optional arguments:

| Arg | Purpose | Default |
|-----|---------|---------|
| 3rd: `[VARIANT]` | Weight variant: empty/bc/distill/... | `bc` (best_model.pth) |
| 4th: `[DES_VEL]` | Desired velocity (m/s) | `5.0` |

```bash
# BC baseline @ 5m/s (default) — uses branch_X/best_model.pth
bash run_full_test.bash D STHMamba
bash run_full_test.bash D STHMamba bc       # explicit

# Distilled model @ 5m/s — uses branch_X/distill_best_model.pth
bash run_full_test.bash D STHMamba distill

# Distilled model @ 7m/s (teacher speed) — matches teacher's flight velocity
bash run_full_test.bash D STHMamba distill 7.0

# Custom variant @ any speed
bash run_full_test.bash D STHMamba distill_from_bc 6.0
```

Results are saved to `results/branch_<X>_<VARIANT>_summary.yaml`.

### Option A: Single Test (Fast, Reusable Simulator)

Use `run_full_test.bash` when the simulator is already running — fastest for sequential branch tests:

```bash
bash run_full_test.bash <BRANCH> <MODEL_TYPE> [VARIANT] [DES_VEL]
```

This script:
- Resets/arms the drone in the existing simulator
- Spawns evaluation_node + run_competition.py
- Polls `start_navigation` until evaluation finishes
- Saves `summary.yaml` to `results/branch_<X>_<variant>_summary.yaml`

### Option B: Full Test (Restarts Simulator Each Run)

Use `launch_mamba_evaluation.bash` for a complete cycle including simulator launch:

```bash
bash launch_mamba_evaluation.bash <N_ROLLOUTS> vision dummy <MODEL_TYPE> <ABSOLUTE_PATH_TO_PTH>
```

Generates `evaluation.yaml` and `envtest/ros/summary.yaml`.

### Option C: Single-Branch Quick Test

```bash
bash test_mamba_branch.bash <BRANCH> <MODEL_TYPE> [VARIANT] [DES_VEL]
```

Self-contained: launches simulator, runs test, kills everything. Slowest but fully isolated.

### Branch Reference

| Branch | Model Type | Architecture | Stateful? |
|--------|-----------|-------------|-----------|
| A | `VMambaLSTM` | VMamba + LSTM | Yes (LSTM hidden state) |
| B | `MambaVisionSSM` | MambaVision + SSM | No |
| B+ | `BPlusModel` | MambaVision + Mamba3 hybrid | No |
| C | `CNNMamba3` | CNN + Mamba3 | No |
| D | `STHMamba` | STH-Mamba | No |
| E | `DecisionMamba` | DecisionMamba | No |

Weights at `experiments/mamba_branches/optimized_training/branch_<X>/`:
- `best_model.pth` — BC baseline (trained via `train_mamba_optimized.py`)
- `distill_best_model.pth` — Distilled model (trained via `train_distill.py`)
- Config files are in each branch directory under `experiments/mamba_branches/`

For the original ViTLSTM baseline: `bash launch_evaluation.bash 1 vision`.

---

## Adding a New Branch (e.g. B+)

When the training pipeline introduces a new architecture, two files need updating:

1. **`envtest/ros/run_competition.py`**:
   - Add model directory to `_BRANCH_MODEL_DIRS`
   - Import the model class in the `try:` block
   - Add an `elif model_type == '<NewType>'` branch in `__init__`

2. **`envtest/ros/user_code.py`**:
   - Add the model class name to `_is_branch_bce` set if it's stateless (returns `(output, None)` and takes 3D velocity)
   - Or add a separate handling block if it has unique input/output requirements

Verify the model loads standalone before testing in simulation:

```python
import torch
sys.path.insert(0, '<branch_models_dir>')
from <model_module> import <ModelClass>
m = <ModelClass>()
ckpt = torch.load('<weight_path>', map_location='cpu')
sd = ckpt.get('model_state_dict', ckpt)
if any(k.startswith('_orig_mod.') for k in sd.keys()):
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
m.load_state_dict(sd)  # ← if this raises, architecture/weight mismatch
```

---

## Diagnosing Model Behavior

Check velocity output lines after a test:

```bash
grep "RUN_COMPETITION.*velocity" /tmp/comp_<X>.log | head -10
wc -l < <(grep "RUN_COMPETITION.*velocity" /tmp/comp_<X>.log)  # output count
```

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| **0 velocity outputs** AND `Success: true` | Model failed to load — drone reached 20m by takeoff inertia alone | Check `/tmp/comp_<X>.log` head for `RuntimeError: Error(s) in loading state_dict` |
| `vy ≈ 0, vz ≈ 0` constant | Mode collapse — model copies input velocity | Retrain with correct target cols 13:16 |
| `vx_out ≈ vx_in` regardless of image | Wrong training target (cols 2:5 used) | See training fix below |
| Extreme `vz` (>1.5) near obstacles | Panic-response model (over-trained, val_loss low but unstable) | Use earlier-epoch weights (early stopping) |
| `Not in hover` repeated | Model never sent commands | Check ROS_MASTER_URI matches running rosmaster |
| `size mismatch for ...` in log | Weight architecture mismatch | Architecture was upgraded; retrain weights |

**False-positive Success warning**: If `summary.yaml` shows `Success: true` but `comp_<X>.log` has 0 velocity outputs, the model **did not run**. Drone reached 20m by takeoff trajectory inertia. Always verify velocity output count matches expected (~240 for 4-second flight at ~60Hz).

**Training target fix** (commit `9152c01`): `train_mamba_optimized.py` must use `traj_meta[idx, 13:16]` (expert velocity command) as target, NOT `traj_meta[idx, 2:5]` (model input). Using cols 2:5 causes identity mapping — model ignores depth images entirely.

---

## Failure Modes & Fixes

### Model Load Failure (architecture/weight mismatch)

Symptom in `/tmp/comp_<X>.log`:
```
RuntimeError: Error(s) in loading state_dict for <ModelClass>:
    size mismatch for ...ss2d.x_proj_weight: copying a param with shape torch.Size([4, 36, 64])
    from checkpoint, the shape in current model is torch.Size([4, 132, 64]).
```

Cause: Model code architecture (e.g. `d_state`, `embed_dim`) was upgraded but weights weren't retrained.

Diagnosis:
```bash
git log --oneline -- experiments/mamba_branches/branch_<X>_*/models/  # find architecture changes
python3 -c "import torch; print(torch.load('<weight_path>')['epoch'])"  # check weight epoch
```

Fix: Either revert architecture to match weights, OR retrain weights with new architecture. Document in commit which is intentional.

### ZMQ "Address already in use"

Ports 10253/10254 held by previous `visionsim_node`:

```bash
# Use safe PID-based kill (not killall/pkill which can hang in WSL2)
for p in roscore rosmaster visionsim_node rviz flight_render; do
    pid=$(ps aux | grep -E "\b$p\b" | grep -v grep | awk '{print $2}' 2>/dev/null)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done
for pid in $(ps aux | grep -E "roslaunch|evaluation_node|run_competition" | grep -v grep | awk '{print $2}' 2>/dev/null); do
    kill -9 $pid 2>/dev/null || true
done
sleep 5
ss -tlnp | grep -E '10253|10254' || echo "ports free"
```

If ports persist: run `wsl --shutdown` from Windows PowerShell, then restart WSL2.

### "Unable to communicate with master"

Most common cause: `ROS_MASTER_URI` defaulted to `http://localhost:11311` instead of `http://192.168.233.250:11311` in the current shell.

```bash
echo $ROS_MASTER_URI                  # verify it's pointing to 192.168.233.250:11311
ss -tlnp | grep 11311                 # check if roscore is listening
```

If env vars wrong → re-export. If master dead → kill everything and restart simulator.

### roscore won't start (stale PID file)

If roscore claims port 11311 is free but still won't start, a stale PID file is blocking it:

```bash
rm -f /root/.ros/roscore-11311.pid
# Then retry: roscore -p 11311 &
```

### `pgrep` / `killall` hangs in WSL2

Some WSL2 environments hang on `pgrep` or `killall` commands (procfs issue). Use `ps aux | grep` instead:

```bash
# ❌ DON'T: pgrep -f roscore     # may hang forever
# ❌ DON'T: killall -9 roscore   # may hang forever

# ✅ DO:
pid=$(ps aux | grep "\<roscore\>" | grep -v grep | awk '{print $2}' 2>/dev/null)
[ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
```

### `visionsim_node` already running — skips relaunch

The launch scripts skip launching if a `visionsim_node` process already exists. If the existing instance lacks RViz/render, kill it first:

```bash
# Use safe PID-based kill (not killall which hangs in WSL2)
for p in visionsim_node rviz flight_render; do
    pid=$(ps aux | grep -E "\b$p\b" | grep -v grep | awk '{print $2}' 2>/dev/null)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done
```

### `run_competition.py` produces no output

Background-launched scripts swallow stderr. Run directly to see errors:

```bash
cd /root/catkin_ws/src/vitfly-mambatest/envtest/ros
python3 -u run_competition.py --vision_based --des_vel 5.0 \
  --model_type CNNMamba3 \
  --model_path /root/catkin_ws/src/vitfly-mambatest/experiments/mamba_branches/optimized_training/branch_C/best_model.pth
```

**Important**: `cd` chained with `&&` may not propagate correctly for background-launched python — use absolute paths or wrap in a separate script.

### Model Dispatch Bug — BPlusModel Falling to Wrong Branch

If B+ (BPlusModel) loads but produces zero velocity output, check `user_code.py`:

**Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x517 and 519x256)`
**Cause**: `_is_branch_bce` set missing `'BPlusModel'` — model falls to `else` branch which passes scalar velocity (1D), but BPlusModel expects 3D velocity (512+3+4=519 vs 512+1+4=517).

```python
# ❌ BUG: BPlusModel not in set, falls to else branch
_is_branch_bce = _class in ('MambaVisionSSMNet', 'CNNMamba3Net', 'STHMambaNet', 'DecisionMambaNet')

# ✅ FIX: include BPlusModel
_is_branch_bce = _class in ('MambaVisionSSMNet', 'CNNMamba3Net', 'STHMambaNet', 'DecisionMambaNet', 'BPlusModel')
```

**Diagnosis**: When adding a new branch model type, ALWAYS check both:
1. `run_competition.py` — model import and `elif model_type == '...'` branch
2. `user_code.py` — `compute_command_vision_based` dispatch logic (`_is_branch_bce` set)

### Triple-Quote Syntax Error

If `run_competition.py` crashes with `SyntaxError: EOF while scanning triple-quoted string literal`:

**Cause**: The upstream `user_code.py` uses `"""..."""` blocks as multi-line comments (dead code examples). When editing the function signature, ensure the `"""` pairs remain balanced.

```python
def compute_command_vision_based(..., seq_len=1):
    """
    Formal docstring here        ← pair 1 opens
    """                          ← pair 1 closes
    """
    # Example of SRT command     ← pair 2 opens (must exist!)
    ...
    """                          ← pair 2 closes
```

The upstream has two `"""` pairs in `compute_command_vision_based` and one in `compute_command_state_based`. When refactoring the docstring, don't remove the example block's opening `"""`.

### `torch.compile()` weight prefix

Weights saved with `torch.compile()` have `_orig_mod.` key prefix. `run_competition.py` strips this automatically. If loading manually:

```python
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```

---

## Multi-Branch Test Strategy

When testing many branches sequentially, persistent simulator is much faster than restarting:

```bash
# Step 1: Launch simulator ONCE
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
sleep 15

# Step 2: Loop through branches with run_full_test.bash
#   - Add "bc" as 3rd arg for BC baseline, "distill" for distilled model
for BRANCH_INFO in "B:MambaVisionSSM" "Bplus:BPlusModel" "C:CNNMamba3" "D:STHMamba" "E:DecisionMamba"; do
  BRANCH="${BRANCH_INFO%%:*}"
  MODEL="${BRANCH_INFO##*:}"
  echo "=== Testing Branch $BRANCH ($MODEL) ==="
  bash run_full_test.bash $BRANCH $MODEL distill   # test distilled weights
done
```

For BC vs Distill comparison, run the loop twice (once with `bc`, once with `distill`).

Saves ~20 seconds per branch (no simulator relaunch).

---

## Quick Reference

```bash
# ⚠ WARNING: killall/pkill can HANG in WSL2. Use targeted PID kills instead:
for p in roscore rosmaster visionsim_node rviz; do
    pid=$(ps aux | grep -E "\b$p\b" | grep -v grep | awk '{print $2}' 2>/dev/null)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done
# Also kill background python processes:
for pid in $(ps aux | grep -E "roslaunch|evaluation_node|run_competition" | grep -v grep | awk '{print $2}' 2>/dev/null); do
    kill -9 $pid 2>/dev/null || true
done
sleep 3

# Remove stale roscore PID file (prevents restart after crash)
rm -f /root/.ros/roscore-11311.pid

# Network setup
ip addr add 192.168.233.250/32 dev lo 2>/dev/null
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME

# Source ROS + conda (note --extend on workspace)
source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash --extend
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH

# Verify ROS is alive
rostopic list | head -5

# Verify raw depth publishes (works after simulator launch)
rostopic echo /kingfisher/dodgeros_pilot/unity/depth --noarr -n 1

# Verify model loads standalone (catches architecture mismatch BEFORE running simulator)
python3 -c "import torch; ckpt=torch.load('<path>', map_location='cpu'); print('epoch:', ckpt.get('epoch','?'), 'val_loss:', ckpt.get('val_loss_gt', ckpt.get('val_loss','?')))"

# Run single branch test (with persistent simulator) — BC baseline @ 5m/s
bash run_full_test.bash D STHMamba

# Run single branch test (with persistent simulator) — distilled model @ 7m/s
bash run_full_test.bash D STHMamba distill 7.0

# Run single branch test (self-contained) @ 5m/s
bash test_mamba_branch.bash D STHMamba distill

# Run single branch test (self-contained) @ 7m/s
bash test_mamba_branch.bash D STHMamba distill 7.0

# Verify model actually ran (not just inertia)
grep "RUN_COMPETITION.*velocity" /tmp/comp_D.log | wc -l   # ~240 for 4s flight

# Check results
cat results/branch_D_distill_summary.yaml

# Compare BC vs Distill side by side (all speeds)
for f in results/branch_*_bc_summary.yaml results/branch_*_distill_summary.yaml; do
    echo "--- $(basename $f) ---"
    grep -E "Success|crash" $f
done

# Eval log (detailed flight info)
cat /tmp/eval_D.log
```
