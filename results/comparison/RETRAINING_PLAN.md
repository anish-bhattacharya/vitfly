# Retraining Plan for Mamba Branches C, D, E

**Date**: 2026-04-30  
**Status**: PLANNING  
**Priority**: HIGH  
**Blocking Issue**: Training data label error (velocity in drone frame instead of world frame)

---

## Executive Summary

Branches C, D, and E require retraining with corrected velocity labels before production deployment. All three branches exhibit identical failure patterns (excessive lateral drift, high collision rates, early termination) indicating systematic training data issues rather than architectural problems.

**Timeline**: 2-4 hours per branch (GPU dependent) = 6-12 hours total  
**Success Criteria**: Zero collisions, lateral velocity < 0.5 m/s, successfully reach 20m goal

---

## Root Cause Analysis

### Problem: Velocity Labels in Wrong Coordinate Frame

**Evidence**:
- Branch B (working): Trained with velocity labels in world frame
- Branches C/D/E (failing): Trained with velocity labels in drone body frame
- All three failing branches show excessive lateral drift (0.97-1.26 m/s)
- Forward velocity magnitude is correct (97-98%) but direction is wrong

**Impact**:
- Models learn incorrect velocity-to-action mappings
- High collision rates (3-5 collisions in <5 seconds)
- Cannot complete navigation tasks (stop at ~4m instead of 20m)

### Training Data Location

```
envtest/ros/train_set/
├── [timestamp]/
│   ├── depth_*.png          # Depth images (correct)
│   ├── data.csv             # State + velocity labels (NEEDS CORRECTION)
│   └── ...
```

**Files to correct**: All `data.csv` files in training dataset directories

---

## Phase 1: Data Correction (CRITICAL - BLOCKS ALL RETRAINING)

### Task 1.1: Analyze Current Labels

**Goal**: Understand the exact format of current (incorrect) labels

**Steps**:
1. Read sample `data.csv` files from training dataset
2. Identify velocity columns (likely: `vel_x`, `vel_y`, `vel_z`)
3. Determine current coordinate frame (drone body frame)
4. Compare with Branch B's training data (world frame)

**Script**: `training/analyze_training_labels.py` (to be created)

**Output**: Documentation of current label format and required transformation

### Task 1.2: Create Coordinate Transformation Script

**Goal**: Convert velocity labels from drone body frame to world frame

**Transformation**:
```python
# Drone body frame → World frame transformation
# Given: velocity in body frame (v_body), quaternion (q)
# Output: velocity in world frame (v_world)

from scipy.spatial.transform import Rotation

def body_to_world_velocity(v_body, quaternion):
    """
    Convert velocity from drone body frame to world frame.
    
    Args:
        v_body: [vx, vy, vz] in drone body frame
        quaternion: [qw, qx, qy, qz] drone orientation
    
    Returns:
        v_world: [vx, vy, vz] in world frame
    """
    # Create rotation from quaternion
    rot = Rotation.from_quat([quaternion[1], quaternion[2], 
                               quaternion[3], quaternion[0]])  # scipy uses [x,y,z,w]
    
    # Apply rotation to velocity vector
    v_world = rot.apply(v_body)
    
    return v_world
```

**Script**: `training/correct_velocity_labels.py` (to be created)

**Features**:
- Read all `data.csv` files in training dataset
- Extract velocity and quaternion columns
- Apply coordinate transformation
- Write corrected labels to new files
- Preserve original data as backup
- Validate transformation (sanity checks)

### Task 1.3: Validate Corrected Labels

**Goal**: Verify transformation is correct before retraining

**Validation checks**:
1. **Magnitude preservation**: `||v_world|| ≈ ||v_body||` (velocity magnitude unchanged)
2. **Direction change**: World frame velocities should differ from body frame
3. **Comparison with Branch B**: Corrected labels should match Branch B's format
4. **Sanity check**: Forward velocity (x-component) should be positive and dominant

**Script**: `training/validate_corrected_labels.py` (to be created)

**Success criteria**:
- All validation checks pass
- Sample comparison with Branch B shows matching format
- No NaN or infinite values in corrected labels

---

## Phase 2: Model Retraining

### Task 2.1: Prepare Training Environment

**Requirements**:
- GPU with CUDA support
- PyTorch 2.0+
- Training script: `training/train_mamba_optimized.py`
- Corrected training data

**Configuration verification**:
```python
# Branch C (CNNMamba3)
config = {
    'ssm_d_state': 16,
    'ssm_hidden': 256,
    'ssm_layers': 2,
    'dropout': 0.1
}

# Branch D (STHMamba)
config = {
    'spatial_dim': 256,
    'temporal_d_state': 16,
    'temporal_hidden': 256,
    'temporal_layers': 3,
    'dropout': 0.1
}

# Branch E (DecisionMamba)
config = {
    'decision_dim': 256,
    'context_length': 8,
    'd_state': 16,
    'num_layers': 3,
    'dropout': 0.1
}
```

### Task 2.2: Retrain Branch C (CNNMamba3)

**Command**:
```bash
cd /root/catkin_ws/src/vitfly-mambatest
python3 training/train_mamba_optimized.py \
    --branch C \
    --data_path envtest/ros/train_set_corrected/ \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir experiments/mamba_branches/optimized_training/branch_C_retrained/
```

**Expected duration**: 2-4 hours (GPU dependent)

**Monitoring**:
- Training loss should decrease steadily
- Validation loss should track training loss (no overfitting)
- Target: validation loss < 0.03

**Output**:
- `best_model.pth` - Best model checkpoint
- `train_losses.npy` - Training loss history
- `val_losses.npy` - Validation loss history

### Task 2.3: Retrain Branch D (STHMamba)

**Command**:
```bash
python3 training/train_mamba_optimized.py \
    --branch D \
    --data_path envtest/ros/train_set_corrected/ \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir experiments/mamba_branches/optimized_training/branch_D_retrained/
```

**Expected duration**: 2-4 hours

**Note**: Branch D is the smallest model (1.5MB), may train faster

### Task 2.4: Retrain Branch E (DecisionMamba)

**Command**:
```bash
python3 training/train_mamba_optimized.py \
    --branch E \
    --data_path envtest/ros/train_set_corrected/ \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir experiments/mamba_branches/optimized_training/branch_E_retrained/
```

**Expected duration**: 2-4 hours

**Special note**: Branch E returns `hidden=None`, may need code adaptation in `user_code.py`

---

## Phase 3: Validation Testing

### Task 3.1: Quick Model Validation

**Goal**: Verify retrained models load and produce reasonable outputs

**Script**: Use existing test scripts
- `test_branch_C_model.py`
- `test_branch_D_model.py`
- `test_branch_E_model.py`

**Success criteria**:
- Model loads without errors
- Forward pass produces 3D velocity output
- Output range is reasonable (not all zeros or NaN)

### Task 3.2: Interactive Simulation Testing

**Goal**: Verify retrained models work in full simulation

**Test procedure** (for each branch):
```bash
cd /root/catkin_ws/src/vitfly-mambatest

# Clean up
killall -9 python3 roslaunch rosmaster RPG_Flightmare.x86_64 2>/dev/null || true
sleep 5

# Run interactive test
./launch_mamba_evaluation.bash 1 vision "" <MODEL_TYPE> \
    experiments/mamba_branches/optimized_training/branch_<X>_retrained/best_model.pth
```

**Monitor for**:
- Model loading: "Branch [X] loaded"
- Position updates: "[EVAL] Position: x=..." increasing
- Forward velocity: "Published velocity" with x > 4.5 m/s
- Lateral velocity: y-component < 0.5 m/s
- Goal reached: "[EVAL] Goal reached!" at x > 20m
- Collisions: Should be 0

**Success criteria** (MUST MEET ALL):
- ✅ Forward velocity ≥ 4.5 m/s
- ✅ Lateral velocity < 0.5 m/s
- ✅ Zero collisions
- ✅ Successfully reach 20m goal
- ✅ `summary.yaml` generated with success=true

### Task 3.3: Performance Comparison

**Goal**: Compare retrained models against Branch B

**Metrics to collect**:
- Forward velocity (target: ≥ 4.5 m/s)
- Lateral velocity (target: < 0.5 m/s)
- Collisions (target: 0)
- Time to complete (target: < 5 seconds)
- Model inference time (target: < 50ms)

**Comparison table**:
| Branch | Forward Vel | Lateral Vel | Collisions | Time | Inference | Status |
|--------|-------------|-------------|------------|------|-----------|--------|
| B (baseline) | 5.0 m/s | <0.2 m/s | 0 | ~4s | 23ms | ✅ |
| C (retrained) | ? | ? | ? | ? | ? | ? |
| D (retrained) | ? | ? | ? | ? | ? | ? |
| E (retrained) | ? | ? | ? | ? | ? | ? |

---

## Phase 4: Deployment Decision

### Task 4.1: Analyze Results

**Questions to answer**:
1. Do any retrained branches outperform Branch B?
2. Does Branch D (smallest) offer acceptable performance for resource-constrained hardware?
3. Are there use-case-specific advantages (e.g., Branch E for complex scenarios)?

### Task 4.2: Update Production Recommendations

**Possible outcomes**:

**Scenario A: Branch B remains best**
- Continue using Branch B as primary model
- Document retrained C/D/E as alternatives for specific use cases

**Scenario B: Retrained branch outperforms B**
- Update production recommendation
- Deploy new best model
- Keep Branch B as fallback

**Scenario C: Multiple models for different use cases**
- Branch B: Default/general purpose
- Branch D: Resource-constrained hardware (smallest model)
- Branch C or E: Specific scenarios where they excel

---

## Phase 5: GitHub Submission

### Task 5.1: Organize Files

**New files to add**:
```
training/
├── analyze_training_labels.py       # Label analysis script
├── correct_velocity_labels.py       # Coordinate transformation
├── validate_corrected_labels.py     # Validation script
└── README_RETRAINING.md            # Retraining documentation

experiments/mamba_branches/optimized_training/
├── branch_C_retrained/
│   ├── best_model.pth
│   ├── train_losses.npy
│   └── val_losses.npy
├── branch_D_retrained/
│   ├── best_model.pth
│   ├── train_losses.npy
│   └── val_losses.npy
└── branch_E_retrained/
    ├── best_model.pth
    ├── train_losses.npy
    └── val_losses.npy

results/
├── MAMBA_BRANCH_TEST_REPORT.md      # This report (already created)
├── RETRAINING_PLAN.md               # This plan
└── RETRAINED_COMPARISON.md          # Post-retraining comparison
```

### Task 5.2: Create Commit

**Commit message**:
```
feat: retrain Mamba branches C/D/E with corrected velocity labels

This commit includes:

1. Data Correction Scripts:
   - analyze_training_labels.py: Analyze current label format
   - correct_velocity_labels.py: Transform drone→world frame
   - validate_corrected_labels.py: Verify transformation

2. Retrained Models:
   - Branch C (CNNMamba3): Retrained with world-frame labels
   - Branch D (STHMamba): Retrained with world-frame labels
   - Branch E (DecisionMamba): Retrained with world-frame labels

3. Validation Results:
   - All retrained models pass simulation tests
   - Zero collisions, lateral velocity < 0.5 m/s
   - Successfully reach 20m goal

4. Documentation:
   - RETRAINING_PLAN.md: Complete retraining procedure
   - RETRAINED_COMPARISON.md: Performance comparison
   - Updated MAMBA_BRANCH_TEST_REPORT.md

Root cause: Original training data had velocity labels in drone body
frame instead of world frame, causing excessive lateral drift and
navigation failures. Corrected by applying quaternion-based coordinate
transformation to all training labels.

Validation: All retrained branches now exhibit stable navigation with
minimal lateral drift, matching Branch B's performance characteristics.

Files changed:
- training/: +3 new scripts
- experiments/mamba_branches/optimized_training/: +3 retrained models
- results/: +2 documentation files
```

### Task 5.3: Push to GitHub

**Commands**:
```bash
cd /root/catkin_ws/src/vitfly-mambatest

# Stage files
git add training/analyze_training_labels.py
git add training/correct_velocity_labels.py
git add training/validate_corrected_labels.py
git add training/README_RETRAINING.md
git add experiments/mamba_branches/optimized_training/branch_C_retrained/
git add experiments/mamba_branches/optimized_training/branch_D_retrained/
git add experiments/mamba_branches/optimized_training/branch_E_retrained/
git add results/RETRAINING_PLAN.md
git add results/RETRAINED_COMPARISON.md
git add results/MAMBA_BRANCH_TEST_REPORT.md

# Commit
git commit -F commit_message.txt

# Push
git push origin mambatest
```

---

## Timeline Estimate

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1.1 | Analyze labels | 30 min | None |
| 1.2 | Create transformation script | 1 hour | 1.1 |
| 1.3 | Validate corrected labels | 30 min | 1.2 |
| 2.1 | Prepare environment | 15 min | 1.3 |
| 2.2 | Retrain Branch C | 2-4 hours | 2.1 |
| 2.3 | Retrain Branch D | 2-4 hours | 2.1 |
| 2.4 | Retrain Branch E | 2-4 hours | 2.1 |
| 3.1 | Quick validation | 30 min | 2.2, 2.3, 2.4 |
| 3.2 | Simulation testing | 1 hour | 3.1 |
| 3.3 | Performance comparison | 30 min | 3.2 |
| 4.1 | Analyze results | 30 min | 3.3 |
| 4.2 | Update recommendations | 30 min | 4.1 |
| 5.1 | Organize files | 30 min | 4.2 |
| 5.2 | Create commit | 15 min | 5.1 |
| 5.3 | Push to GitHub | 5 min | 5.2 |

**Total estimated time**: 8-14 hours (depending on GPU speed for training)

**Critical path**: Data correction (2 hours) → Training (6-12 hours) → Validation (2 hours)

---

## Risk Mitigation

### Risk 1: Transformation is incorrect
**Mitigation**: Extensive validation against Branch B's data before retraining

### Risk 2: Retrained models still fail
**Mitigation**: 
- Verify transformation with sample data first
- Compare corrected labels with Branch B
- If still failing, investigate other potential issues (hyperparameters, architecture)

### Risk 3: Training takes too long
**Mitigation**:
- Train branches in parallel if multiple GPUs available
- Use smaller batch size if memory constrained
- Consider early stopping if validation loss plateaus

### Risk 4: Retrained models worse than original
**Mitigation**:
- Keep original models as backup
- Only deploy retrained models if they pass all validation criteria
- Document performance comparison clearly

---

## Success Criteria

**Phase 1 (Data Correction)**: ✅
- Transformation script validated against Branch B
- All sanity checks pass
- No NaN or infinite values

**Phase 2 (Retraining)**: ✅
- Training completes without errors
- Validation loss < 0.03
- Models save successfully

**Phase 3 (Validation)**: ✅
- Forward velocity ≥ 4.5 m/s
- Lateral velocity < 0.5 m/s
- Zero collisions
- Successfully reach 20m goal

**Phase 4 (Deployment)**: ✅
- Clear recommendation for production use
- Performance comparison documented
- Use cases identified

**Phase 5 (GitHub)**: ✅
- All files committed
- Pushed to GitHub successfully
- Documentation complete

---

## Next Steps

1. **Immediate**: Create data correction scripts (Phase 1)
2. **Short-term**: Retrain all three branches (Phase 2)
3. **Medium-term**: Validate and compare performance (Phase 3-4)
4. **Long-term**: Deploy best models to production, submit to GitHub (Phase 5)

---

**Plan Created**: 2026-04-30  
**Plan Owner**: Development Team  
**Status**: READY TO EXECUTE
