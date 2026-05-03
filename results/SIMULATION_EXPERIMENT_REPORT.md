# Vitfly Mamba Drone Obstacle Avoidance Simulation Experiment Report

**Date**: 2026-05-03  
**Test Environment**: WSL2 + ROS Noetic + Flightmare + Unity  
**Branches Tested**: 6 (A, B, B+, C, D, E)  
**Test Outcome**: 5/6 Pass, 1/6 Fail

---

## 1. Executive Summary

This session completed comprehensive simulation testing of six Mamba-based drone obstacle avoidance architectures using the Flightmare/Unity simulation environment. Five branches (B, B+, C, D, E) successfully completed the 20-meter obstacle course in approximately 4.2 seconds with zero collisions. Branch A failed due to an architecture/weight mismatch where the model code was upgraded to `d_state=64` but weights remained at `d_state=16`. Critical discoveries include: (1) RViz depth visualization requires `run_competition.py` to be active, not just the simulator; (2) false-positive success detection when models fail to load but drones reach goals via takeoff inertia; (3) post-goal collisions are not counted as failures; (4) persistent simulator approach reduces testing time by ~20 seconds per branch. These findings have been integrated into the vitfly skill documentation and inform recommendations for the training pipeline.

---

## 2. Test Results

| Branch | Model | Result | Crashes | Time (s) | Velocity Outputs | Notes |
|--------|-------|--------|---------|----------|------------------|-------|
| **A** | MambaCSM | ❌ **FAIL** | N/A | N/A | 0 | Architecture/weight mismatch: d_state=16 weights vs d_state=64 model |
| B | MambaVisionSSM | ✅ PASS | 0 | 4.20 | ~240 | Clean flight, stable outputs |
| B+ | BPlusModel | ✅ PASS | 0 | 4.18 | ~240 | Newly added architecture, successful |
| C | CNNMamba3 | ✅ PASS | 0 | 4.22 | ~240 | Hybrid CNN-Mamba approach |
| D | STHMamba | ✅ PASS | 0 | 4.19 | ~240 | Spatial-temporal hybrid |
| E | DecisionMamba | ✅ PASS | 0 | 4.21 | ~240 | Decision-focused architecture |

**Result Files**: `results/branch_<X>_full_summary.yaml`

**Branch A Root Cause**: Commit `d625320` upgraded model architecture from `d_state=16` to `d_state=64`, but training weights were not regenerated. The `state_dict` shape mismatch (`torch.Size([4, 36, 64])` in checkpoint vs `torch.Size([4, 132, 64])` in model) prevented model loading. The drone appeared to succeed by reaching 20m via takeoff inertia alone, but velocity output count of 0 confirmed the model never ran.

---

## 3. Testing Infrastructure

### Environment
- **OS**: WSL2 (Ubuntu 20.04)
- **ROS**: Noetic
- **Simulator**: Flightmare + Unity (Flightmare_Env.x86_64)
- **Visualization**: RViz
- **Network**: Fixed IP `192.168.233.250` (not dynamic hostname)

### Network Setup
```bash
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
```

**Critical**: Do NOT use `hostname -I` for dynamic IP detection. The simulator and ROS master require consistent fixed IP configuration.

### Three Test Methods

| Method | Script | Simulator Lifecycle | Speed | Use Case |
|--------|--------|---------------------|-------|----------|
| **Persistent** | `run_full_test.bash` | External (persistent) | Fastest | Sequential multi-branch testing |
| **Full Cycle** | `launch_mamba_evaluation.bash` | Managed (restart each test) | Medium | Single branch with clean state |
| **Self-Contained** | `test_mamba_branch.bash` | Managed (full lifecycle) | Slowest | Isolated testing, debugging |

**Recommendation**: Use `run_full_test.bash` with persistent simulator for batch testing (saves ~20s per branch).

### Files Created/Modified

**Test Infrastructure**:
- `run_full_test.bash` - Persistent simulator test runner
- `launch_mamba_evaluation.bash` - Full-cycle test with simulator restart
- `test_mamba_branch.bash` - Self-contained test script

**Results**:
- `results/branch_A_full_summary.yaml` - Branch A failure details
- `results/branch_B_full_summary.yaml` - Branch B success metrics
- `results/branch_Bplus_full_summary.yaml` - Branch B+ success metrics
- `results/branch_C_full_summary.yaml` - Branch C success metrics
- `results/branch_D_full_summary.yaml` - Branch D success metrics
- `results/branch_E_full_summary.yaml` - Branch E success metrics

**Documentation**:
- `.claude/skills/vitfly/SKILL.md` - Updated with 10 key diagnostic improvements
- `results/SIMULATION_EXPERIMENT_REPORT.md` - This report

---

## 4. Diagnostic Experiences (Critical Section)

### 4.1 RViz Depth Image Display

**Misconception**: "RViz isn't showing depth images between tests, so the simulator is broken."

**Reality**: Two separate depth topics exist:
- `/kingfisher/dodgeros_pilot/unity/depth` - Raw depth data (always publishing after simulator launch)
- `/kingfisher/dodgeros_pilot/unity/depth_viz` - Visualization topic (only publishes when `run_competition.py` is active)

**RViz subscribes to `depth_viz`** by default, so an empty RViz display between test runs is normal and expected behavior.

**Verification Command**:
```bash
# Check raw depth is publishing
rostopic echo /kingfisher/dodgeros_pilot/unity/depth --noarr -n 1

# Expected output: header with timestamp
# If timeout: simulator not running or network misconfigured
```

**Key Insight**: Empty RViz ≠ broken simulator. Always verify raw depth topic, not visualization.

---

### 4.2 False-Positive Success Detection

**Symptom**: `summary.yaml` shows `Success: true, 0 crashes` but model clearly didn't run.

**Root Cause**: The drone's takeoff trajectory provides enough forward velocity to reach the 20-meter goal line by inertia alone, even if the obstacle avoidance model fails to load.

**Detection Method**: Check velocity output count in competition log:
```bash
grep "RUN_COMPETITION.*velocity" /tmp/comp_<X>.log | wc -l

# Expected: ~240 outputs for 4-second flight at 60Hz
# If 0: model failed to load, "success" is false positive
```

**Branch A Example**:
- `summary.yaml`: `Success: true, Crashes: 0, Time: 4.2s`
- Velocity outputs: **0**
- Actual status: Model load failure masked by takeoff inertia

**Always Verify**: Cross-reference `Success: true` with velocity output count. Zero outputs = model didn't run.

---

### 4.3 Architecture/Weight Mismatch

**Symptom**: `RuntimeError: Error(s) in loading state_dict` with shape mismatch errors:
```
size mismatch for vmamba.blocks.0.ss2d.x_proj_weight:
  copying a param with shape torch.Size([4, 36, 64]) from checkpoint,
  the shape in current model is torch.Size([4, 132, 64])
```

**Root Cause**: Model architecture was upgraded (e.g., `d_state=16` → `d_state=64`) but training weights were not regenerated to match the new architecture.

**Branch A Case Study**:
- Commit `d625320`: Upgraded architecture to `d_state=64`
- Weights: Still from `d_state=16` training
- Result: Incompatible shapes prevent model loading

**Diagnosis Steps**:
1. **Check git history for architecture changes**:
   ```bash
   git log --oneline -- experiments/mamba_branches/branch_A_*/models/
   ```

2. **Verify weight training epoch**:
   ```python
   import torch
   ckpt = torch.load('path/to/weights.pth', map_location='cpu')
   print(f"Epoch: {ckpt.get('epoch', 'unknown')}")
   print(f"Keys: {list(ckpt.keys())}")
   ```

3. **Standalone load test** (before simulation):
   ```python
   import torch, sys
   sys.path.insert(0, 'experiments/mamba_branches/branch_A_mamba_csm/models')
   from mamba_csm import MambaCSM
   
   model = MambaCSM()
   ckpt = torch.load('training/checkpoints/branch_A/best_model.pth', map_location='cpu')
   state_dict = ckpt.get('model_state_dict', ckpt)
   
   # Handle torch.compile artifacts
   if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
       state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
   
   model.load_state_dict(state_dict)  # ← Catches mismatch before simulation
   print("✓ Model loaded successfully")
   ```

**Fix Options**:
- **Option 1** (Recommended): Retrain weights with new architecture
- **Option 2**: Revert architecture to match existing weights
- **Option 3**: Keep old weights as `_legacy.pth` for ablation studies

---

### 4.4 Post-Goal Collisions

**Observation**: In RViz, the drone collides with an obstacle immediately after crossing the 20-meter goal line.

**Explanation**: The `evaluation_node.py` stops monitoring collisions once the drone reaches the 20-meter segment. The drone continues moving by inertia, and any post-goal collisions are not counted.

**Not a Failure**: A result of `Success: true, 0 crashes` is correct even if RViz shows a collision after the goal.

**Why This Happens**:
- Evaluation window: `[0m, 20m]`
- Drone velocity at 20m: ~1.0 m/s forward
- Post-goal trajectory: Uncontrolled (model stops running)
- Collision timing: After evaluation ends

**Key Insight**: Only collisions within the 20-meter evaluation segment count as failures.

---

### 4.5 Environment Variable Persistence

**Problem**: Manual `rostopic` or `roslaunch` commands fail with "Unable to communicate with master" even though test scripts work fine.

**Root Cause**: Default shell sessions have `ROS_MASTER_URI=http://localhost:11311`, not the required `http://192.168.233.250:11311`.

**Solution**: Export environment variables in **every** shell that runs ROS commands:
```bash
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
```

**Test Scripts Handle This Internally**: The `run_full_test.bash` and related scripts set these variables automatically. This issue only affects manual debugging commands.

**Verification**:
```bash
# Check current ROS configuration
echo $ROS_MASTER_URI
echo $ROS_IP
echo $ROS_HOSTNAME

# Test connectivity
rostopic list
```

---

## 5. Multi-Branch Testing Strategy

### Persistent Simulator Approach (Recommended)

For sequential testing of multiple branches, launch the simulator once and reuse it:

```bash
# Step 1: Launch simulator ONCE
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
sleep 15  # Wait for Unity to initialize

# Step 2: Loop through branches
for BRANCH_INFO in "B:MambaVisionSSM" "Bplus:BPlusModel" "C:CNNMamba3" "D:STHMamba" "E:DecisionMamba"; do
  BRANCH="${BRANCH_INFO%%:*}"
  MODEL="${BRANCH_INFO##*:}"
  
  echo "Testing Branch $BRANCH with model $MODEL..."
  bash run_full_test.bash $BRANCH $MODEL
  
  sleep 5  # Brief pause between tests
done

# Step 3: Cleanup
rosnode kill -a
killall Flightmare_Env.x86_64
```

**Advantages**:
- Saves ~20 seconds per branch (no simulator relaunch)
- Consistent environment across all tests
- Faster iteration during development

**Disadvantages**:
- Simulator state persists between tests (usually not an issue)
- Single simulator crash affects all remaining tests

**When to Use**: Batch testing of multiple branches, regression testing, CI/CD pipelines.

---

## 6. Adding New Branches

When the training pipeline introduces a new architecture (e.g., Branch B+), follow this checklist:

### Step 1: Update `envtest/ros/run_competition.py`

Add model directory and import:
```python
_BRANCH_MODEL_DIRS = {
    'A': 'experiments/mamba_branches/branch_A_mamba_csm/models',
    'B': 'experiments/mamba_branches/branch_B_mambavision_ssm/models',
    'Bplus': 'experiments/mamba_branches/branch_Bplus_optimized/models',  # ← New
    # ...
}

# In imports section
try:
    from mamba_csm import MambaCSM
    from mamba_vision_ssm import MambaVisionSSM
    from bplus_model import BPlusModel  # ← New
    # ...
except ImportError as e:
    rospy.logwarn(f"Model import failed: {e}")
```

Add model instantiation:
```python
def __init__(self):
    # ...
    if model_type == 'MambaCSM':
        self.model = MambaCSM()
    elif model_type == 'MambaVisionSSM':
        self.model = MambaVisionSSM()
    elif model_type == 'BPlusModel':  # ← New
        self.model = BPlusModel()
    # ...
```

### Step 2: Update `envtest/ros/user_code.py`

Add model class name to appropriate category:
```python
# For stateless models (no hidden state)
_is_branch_bce = {
    'MambaVisionSSM',
    'BPlusModel',  # ← New
    'CNNMamba3',
    'STHMamba',
    'DecisionMamba'
}
```

Or add separate handling if the model has unique input/output requirements:
```python
elif isinstance(self.model, BPlusModel):
    # Custom preprocessing
    depth_processed = custom_preprocess(depth_image)
    output = self.model(depth_processed)
    # Custom postprocessing
    velocity = custom_postprocess(output)
```

### Step 3: Verify Standalone Load

Before running simulation, test model loading in isolation:
```python
import torch, sys
sys.path.insert(0, 'experiments/mamba_branches/branch_Bplus_optimized/models')
from bplus_model import BPlusModel

model = BPlusModel()
ckpt = torch.load('training/checkpoints/branch_Bplus/best_model.pth', map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

# Handle torch.compile artifacts
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
print("✓ Model loaded successfully")

# Test forward pass
dummy_input = torch.randn(1, 1, 60, 80)
output = model(dummy_input)
print(f"Output shape: {output.shape}")
```

### Step 4: Run Test

```bash
bash run_full_test.bash Bplus BPlusModel
```

---

## 7. Vitfly Skill Updates

The `.claude/skills/vitfly/SKILL.md` was updated with 10 key improvements based on this testing session:

1. **RViz Depth Visualization Mechanism**: Documented the distinction between `/unity/depth` (raw, always publishing) and `/unity/depth_viz` (visualization, only when `run_competition.py` is active). Empty RViz between tests is normal.

2. **False-Positive Success Detection**: Added velocity output count verification method to catch model load failures masked by takeoff inertia.

3. **Architecture/Weight Mismatch Diagnosis**: Comprehensive guide for diagnosing `state_dict` shape errors, including git history checks and standalone load testing.

4. **Post-Goal Collision Explanation**: Clarified that collisions after the 20-meter goal are not counted as failures.

5. **Branch B+ Reference**: Added Branch B+ to the branch reference table with model class `BPlusModel` and architecture notes.

6. **New Branch Addition Checklist**: Step-by-step guide for integrating new architectures into the testing pipeline.

7. **Three Test Method Comparison**: Documented trade-offs between `run_full_test.bash`, `launch_mamba_evaluation.bash`, and `test_mamba_branch.bash`.

8. **Multi-Branch Persistent-Simulator Strategy**: Added bash loop example for efficient sequential testing.

9. **Standalone Model Load Verification**: Python snippet for pre-simulation model loading tests to catch architecture issues early.

10. **Environment Variable Persistence Warning**: Documented the need to export `ROS_MASTER_URI`/`ROS_IP` in every shell for manual commands.

---

## 8. Recommendations for Training Pipeline

### 8.1 Branch A Retraining (Highest Priority)

**Current State**: Model architecture uses `d_state=64`, but weights are from `d_state=16` training (incompatible).

**Action Required**: Retrain Branch A from scratch with the upgraded architecture.

**Timeline**: [Requires training team input]

**Backup Strategy**: Archive old `d_state=16` weights as `branch_A_d_state16_legacy.pth` for ablation studies comparing old vs new architecture.

---

### 8.2 Validation Loss vs Real Performance Disconnect

**Observation**:
- Branch B epoch 1: `val_loss=0.0274` → Clean flight ✅
- Branch B epoch 4: `val_loss=0.0205` → Success but extreme `vz=2.98` (panic response)

**Issue**: Lower validation loss does not guarantee better real-world behavior. MSE loss alone doesn't capture output distribution quality.

**Recommendation**: Track per-dimension output statistics during training:
```python
# In training loop
vy_std = torch.std(predictions[:, 1]).item()
vz_std = torch.std(predictions[:, 2]).item()
vz_max = torch.max(torch.abs(predictions[:, 2])).item()

wandb.log({
    'val_loss': loss,
    'vy_std': vy_std,
    'vz_std': vz_std,
    'vz_max': vz_max
})

# Flag models with extreme outputs
if vz_max > 1.5:
    print(f"⚠️  Epoch {epoch}: Panic response detected (vz_max={vz_max:.2f})")
```

**Benefit**: Early detection of mode collapse or panic-response behaviors before simulation testing.

---

### 8.3 Output Variance Monitoring

**Historical Context**: Previous training runs produced near-constant velocity outputs (mode collapse), where models learned to output safe but non-reactive commands.

**Current Status**: Epoch-1 training successfully escapes mode collapse, but no explicit guard exists to detect regression.

**Recommendation**: Add output variance as a standard training metric:
```python
# Per-epoch variance check
vy_variance = torch.var(predictions[:, 1]).item()
vz_variance = torch.var(predictions[:, 2]).item()

if vy_variance < 0.01 or vz_variance < 0.01:
    print(f"⚠️  Low output variance detected - possible mode collapse")
    print(f"   vy_var={vy_variance:.4f}, vz_var={vz_variance:.4f}")
```

**Threshold Suggestion**: Flag models with `std(vy) < 0.1` or `std(vz) < 0.1` as potentially collapsed.

---

### 8.4 Per-Branch Architecture Notes

**Current Landscape**: Six distinct architectures with different state management:

| Branch | Architecture | State Management | Notes |
|--------|--------------|------------------|-------|
| A | MambaCSM | Stateful (LSTM hidden state) | Requires state reset between episodes |
| B | MambaVisionSSM | Stateless | Direct depth → velocity mapping |
| B+ | BPlusModel | Stateless | Optimized variant of Branch B |
| C | CNNMamba3 | Stateless | Hybrid CNN-Mamba approach |
| D | STHMamba | Stateless | Spatial-temporal hybrid |
| E | DecisionMamba | Stateless | Decision-focused architecture |

**Observation**: Branch A's stateful nature may explain different failure modes compared to stateless branches.

**Recommendation**: Document state management requirements in model docstrings:
```python
class MambaCSM(nn.Module):
    """
    Mamba-based obstacle avoidance model with LSTM hidden state.
    
    State Management: STATEFUL
    - Requires hidden state reset at episode start
    - Hidden state shape: (1, 1, 128)
    - Call model.reset_hidden() before each episode
    """
```

---

### 8.5 Multi-Environment Testing (Future Work)

**Current Limitation**: All tests run on `environment_0` of the `spheres_medium` configuration.

**Available Resources**: 100+ pre-generated environments in `flightmare/flightpy/configs/vision/spheres_medium/`.

**Recommendation**: Implement multi-environment robustness testing:

1. **Select Representative Environments**:
   - `environment_0`: Baseline (current)
   - `environment_25`: Low obstacle density
   - `environment_50`: Medium obstacle density
   - `environment_75`: High obstacle density
   - `environment_99`: Maximum difficulty

2. **Report Success Rate**: `X/5 environments passed` per branch

3. **Implementation**:
   ```bash
   for ENV_ID in 0 25 50 75 99; do
     # Modify config to use environment_$ENV_ID
     sed -i "s/environment_[0-9]\\+/environment_$ENV_ID/" \
       flightmare/flightpy/configs/vision/config.yaml
     
     bash run_full_test.bash B MambaVisionSSM
     mv results/branch_B_full_summary.yaml \
       results/branch_B_env${ENV_ID}_summary.yaml
   done
   ```

**Benefit**: Reveals overfitting to single environment, validates generalization.

---

### 8.6 Test Protocol Documentation

**Current State**: Test procedures are scattered across bash scripts and tribal knowledge.

**Recommendation**: Centralize test protocol documentation in `.claude/skills/vitfly/SKILL.md` (already completed in this session).

**Key Components**:
- Unity/RViz lifecycle management
- Drone reset procedures
- Summary YAML generation
- Diagnostic troubleshooting guide

**Benefit**: Onboarding new team members, reproducible testing, CI/CD integration.

---

## 9. Quick Reference Commands

### Full Clean Restart
```bash
# Kill all ROS nodes and simulator
rosnode kill -a
killall Flightmare_Env.x86_64
killall rviz

# Set environment variables
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME

# Launch fresh simulator
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True &
sleep 15
```

### Network Setup Verification
```bash
# Check environment variables
echo "ROS_MASTER_URI: $ROS_MASTER_URI"
echo "ROS_IP: $ROS_IP"
echo "ROS_HOSTNAME: $ROS_HOSTNAME"

# Expected output:
# ROS_MASTER_URI: http://192.168.233.250:11311
# ROS_IP: 192.168.233.250
# ROS_HOSTNAME: (empty)
```

### Verify ROS is Alive
```bash
rostopic list | head -5

# Expected: List of topics including /kingfisher/...
# If error: Check ROS_MASTER_URI and network configuration
```

### Verify Raw Depth Publishes
```bash
rostopic echo /kingfisher/dodgeros_pilot/unity/depth --noarr -n 1

# Expected: header with timestamp
# If timeout: Simulator not running or network issue
```

### Verify Model Loads Standalone
```python
import torch, sys
sys.path.insert(0, 'experiments/mamba_branches/branch_B_mambavision_ssm/models')
from mamba_vision_ssm import MambaVisionSSM

model = MambaVisionSSM()
ckpt = torch.load('training/checkpoints/branch_B/best_model.pth', map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
print("✓ Model loaded successfully")
```

### Run Single Branch Test (3 Methods)

**Method 1: Persistent Simulator** (fastest)
```bash
bash run_full_test.bash B MambaVisionSSM
```

**Method 2: Full Cycle** (medium)
```bash
bash launch_mamba_evaluation.bash B MambaVisionSSM
```

**Method 3: Self-Contained** (slowest, most isolated)
```bash
bash test_mamba_branch.bash B MambaVisionSSM
```

### Verify Model Actually Ran
```bash
# Check velocity output count
grep "RUN_COMPETITION.*velocity" /tmp/comp_B.log | wc -l

# Expected: ~240 for 4-second flight at 60Hz
# If 0: Model failed to load (false-positive success)
```

### Check Results
```bash
# View summary
cat results/branch_B_full_summary.yaml

# Check for success
grep "Success" results/branch_B_full_summary.yaml

# Check crash count
grep "Crashes" results/branch_B_full_summary.yaml

# View full competition log
less /tmp/comp_B.log

# View evaluation log
less /tmp/eval_B.log
```

---

## 10. Artifacts

### Per-Branch Summaries
- `results/branch_A_full_summary.yaml` - Branch A failure (architecture mismatch)
- `results/branch_B_full_summary.yaml` - Branch B success
- `results/branch_Bplus_full_summary.yaml` - Branch B+ success
- `results/branch_C_full_summary.yaml` - Branch C success
- `results/branch_D_full_summary.yaml` - Branch D success
- `results/branch_E_full_summary.yaml` - Branch E success

### Test Infrastructure
- `run_full_test.bash` - Persistent simulator test runner (recommended)
- `launch_mamba_evaluation.bash` - Full-cycle test with simulator restart
- `test_mamba_branch.bash` - Self-contained test script

### Logs
- `/tmp/comp_<X>.log` - Competition node output (velocity commands, model status)
- `/tmp/eval_<X>.log` - Evaluation node output (collision detection, success criteria)

### Documentation
- `.claude/skills/vitfly/SKILL.md` - Updated with 10 diagnostic improvements
- `results/SIMULATION_EXPERIMENT_REPORT.md` - This comprehensive report

---

## Conclusion

This testing session successfully validated five of six Mamba-based obstacle avoidance architectures in simulation. The Branch A failure revealed critical gaps in the architecture-weight synchronization process, leading to improved diagnostic procedures now documented in the vitfly skill. Key operational discoveries—including RViz depth visualization behavior, false-positive success detection, and persistent simulator optimization—have been integrated into the testing infrastructure and will accelerate future development cycles.

**Next Steps**:
1. Training team: Retrain Branch A with `d_state=64` architecture
2. Testing team: Implement multi-environment robustness testing (5 environments)
3. All teams: Adopt velocity output count verification as standard practice
4. Training team: Add output variance monitoring to training metrics

**Contact**: For questions about this report or testing procedures, refer to `.claude/skills/vitfly/SKILL.md` or consult the testing team.
