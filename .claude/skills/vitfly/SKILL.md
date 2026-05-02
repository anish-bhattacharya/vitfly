---
name: vitfly
description: "Operational runbook for ROS1 Noetic + WSL2 + Flightmare simulation testing of Mamba drone obstacle avoidance models. Use when launching simulations, diagnosing failures, or running branch evaluations."
origin: local
---

# Vitfly Simulation Runbook

## When to Use

- Starting a simulation test session after WSL2 restart
- Diagnosing ZMQ port conflicts, IP mismatches, or ROS master failures
- Running evaluation tests for Mamba branches (A–E)
- Diagnosing model mode collapse or unexpected flight behavior

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

Required environment variables:

```bash
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
export FLIGHTMARE_PATH=/root/catkin_ws/src/vitfly/flightmare
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
```

**Critical**: Always use the fixed IP `192.168.233.250`, never `$(hostname -I)`. The dynamic IP causes ROS master connection failures when the loopback alias is set to the fixed IP.

---

## Launch Sequence

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
```

Launch simulator (roscore + Unity renderer + RViz in one command):

```bash
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
sleep 15  # wait for Unity ZMQ connection
```

Reset and arm drone:

```bash
rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once
```

---

## Model Testing Workflow

Use `test_mamba_branch.bash` (project root) to test a single branch:

```bash
bash test_mamba_branch.bash <BRANCH> <MODEL_TYPE>
```

Branch reference:

| Branch | Model Type | Architecture |
|--------|-----------|-------------|
| A | `VMambaLSTM` | VMamba + LSTM (stateful) |
| B | `MambaVisionSSM` | MambaVision + SSM |
| C | `CNNMamba3` | CNN + Mamba3 |
| D | `STHMamba` | STH-Mamba |
| E | `DecisionMamba` | DecisionMamba |

Weights are at `experiments/mamba_branches/optimized_training/branch_<X>/best_model.pth`.

Results saved to `results/branch_<X>_epoch1_summary.yaml` and `evaluation.yaml`.

For the original ViTLSTM baseline: `bash launch_evaluation.bash 1 vision`.

---

## Diagnosing Model Behavior

Check velocity output lines after a test:

```bash
grep "RUN_COMPETITION.*velocity" /tmp/branch_B_epoch1.log | head -10
```

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `vy ≈ 0, vz ≈ 0` constant | Mode collapse — model copies input velocity | Retrain with correct target cols 13:16 |
| `vx_out ≈ vx_in` regardless of image | Wrong training target (cols 2:5 used) | See training fix below |
| Zero velocity lines in log | `run_competition.py` crashed silently | Run directly to see error |
| `Not in hover` repeated | Model never sent commands | Check ROS_MASTER_URI matches running rosmaster |

**Training target fix** (commit `9152c01`): `train_mamba_optimized.py` must use `traj_meta[idx, 13:16]` (expert velocity command) as target, NOT `traj_meta[idx, 2:5]` (model input). Using cols 2:5 causes identity mapping — model ignores depth images entirely.

---

## Failure Modes & Fixes

### ZMQ "Address already in use"

Ports 10253/10254 held by previous `visionsim_node`:

```bash
killall -9 roscore rosmaster rosout visionsim_node rviz flight_render
pkill -9 -f "roslaunch|evaluation_node|run_competition"
sleep 5
ss -tlnp | grep -E '10253|10254' || echo "ports free"
```

If ports persist: run `wsl --shutdown` from Windows PowerShell, then restart WSL2.

### "Unable to communicate with master"

```bash
pgrep -a rosmaster          # check if running
ss -tlnp | grep 11311       # check which IP it's on
```

Kill everything and restart with fixed IP `192.168.233.250`.

### `visionsim_node` already running — skips relaunch

The launch scripts skip launching if `pgrep visionsim_node` finds a process. If the existing instance lacks RViz/render, kill it first:

```bash
killall -9 visionsim_node rviz flight_render
```

### `run_competition.py` produces no output

Run directly to see errors (the launch script swallows stderr):

```bash
cd envtest/ros
python3 -u run_competition.py --vision_based --des_vel 5.0 \
  --model_type CNNMamba3 \
  --model_path /root/catkin_ws/src/vitfly-mambatest/experiments/mamba_branches/optimized_training/branch_C/best_model.pth
```

### `torch.compile()` weight prefix

Weights saved with `torch.compile()` have `_orig_mod.` key prefix. `run_competition.py` strips this automatically. If loading manually:

```python
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```

---

## Quick Reference

```bash
# Full clean restart
killall -9 roscore rosmaster rosout visionsim_node rviz flight_render 2>/dev/null
pkill -9 -f "roslaunch|evaluation_node|run_competition" 2>/dev/null
sleep 5

# Network setup
ip addr add 192.168.233.250/32 dev lo 2>/dev/null
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250

# Check ROS is alive
rostopic list | head -5

# Check depth images publishing
rostopic hz /kingfisher/dodgeros_pilot/unity/depth --window=3

# Run single branch test
bash test_mamba_branch.bash D STHMamba

# Check results
cat evaluation.yaml
grep "RUN_COMPETITION.*velocity" /tmp/branch_D_epoch1.log | wc -l
```
