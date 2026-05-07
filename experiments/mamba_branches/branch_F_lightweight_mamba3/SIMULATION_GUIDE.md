# Branch F Simulation Testing Guide

## Pipeline Overview

Branch F training and simulation are **hard-isolated** across two environments:

| Environment | Host | Purpose | Location |
|-------------|------|---------|----------|
| **Training** | AutoDL (GPU server) | Model training & checkpoint export | `/root/vitfly/training/` |
| **Simulation** | WSL2 + ROS Noetic + Flightmare | Evaluation in realistic quadrotor simulation | `/root/catkin_ws/src/vitfly-mambatest/` |

### Data Flow

```
AutoDL (Training)
    │
    ├── python train_mamba_optimized.py --branches F
    │   └── Checkpoints saved to:
    │       experiments/mamba_branches/branch_F_lightweight_mamba3/checkpoints/
    │
    ├── bash training/deploy_branch_f_checkpoints.sh
    │   └── Copies to standardized location → git commit + push
    │
    ▼
GitHub (origin/mambatest)
    │
    ├── git pull (on simulation machine)
    │
    ▼
WSL2 + ROS (Simulation)
    │
    ├── bash run_full_test.bash F BranchFModel bc
    ├── bash run_full_test.bash F BranchFModel bc_aug
    └── bash run_full_test.bash F BranchFModel multiseq
```

## Training Variants

### F1-BC — Pure Behavioral Cloning Baseline

- **Description**: No augmentation, no distillation, single-frame input
- **Training config**: `training/configs/branch_f_bc_baseline.yaml`
- **Target**: Establish architecture baseline — can a 1.1M param CNN+Mamba-3 match a 3.56M ViT+LSTM?
- **Priority**: P0 (prerequisite for all other experiments)

### F2-BC-Aug — BC with Data Augmentation

- **Description**: Adds horizontal flip (p=0.5) + depth noise (σ=0.02, p=0.3)
- **Training config**: Default config with augmentation flags enabled
- **Target**: Measure isolated effect of data augmentation
- **Priority**: P1 (depends on F1-BC completion)

### F4-MultiSeq — Multi-Sequence Temporal

- **Description**: Multi-frame (seq_len=8) with Mamba-3 SSM recurrence
- **Training config**: Multi-frame dataloader (`create_sequence_dataloader`)
- **Target**: Tests temporal SSM capabilities beyond single-frame
- **Priority**: P1 (independent of F2/F3)

> **Note**: F3-Distill (Progressive Hierarchical Distillation) is implemented but its deployment is handled separately via the distillation pipeline (`train_branch_f_distill.py`).

## Checkpoint Locations

### Source (training outputs)

Located under `experiments/mamba_branches/branch_F_lightweight_mamba3/checkpoints/`:

| Variant | Source Path |
|---------|-------------|
| F1-BC | `f1_bc_baseline/branch_F/best_model.pth` |
| F2-BC-Aug | `f2_bc_aug/branch_F/best_model.pth` |
| F4-MultiSeq | `f4_multiseq/branch_F/best_model.pth` |

### Deployment (standardized for simulation)

Located under `experiments/mamba_branches/optimized_training/branch_F/`:

| Variant | Deployed As | Used By |
|---------|-------------|---------|
| F1-BC | `best_model.pth` | `run_full_test.bash F BranchFModel bc` |
| F2-BC-Aug | `bc_aug_best_model.pth` | `run_full_test.bash F BranchFModel bc_aug` |
| F4-MultiSeq | `multiseq_best_model.pth` | `run_full_test.bash F BranchFModel multiseq` |

### Naming Convention

The `run_full_test.bash` script resolves model paths as follows:
- **bc** (or empty): `${DEPLOY_DIR}/best_model.pth`
- **bc_aug**: `${DEPLOY_DIR}/bc_aug_best_model.pth`
- **multiseq**: `${DEPLOY_DIR}/multiseq_best_model.pth`

## Deployment (AutoDL side)

After all three trainings complete, deploy from the **training machine**:

```bash
cd /root/vitfly
bash training/deploy_branch_f_checkpoints.sh
```

This will:
1. Verify all three checkpoints exist and are valid
2. Copy to `experiments/mamba_branches/optimized_training/branch_F/`
3. Back up any existing checkpoints to a timestamped subdirectory
4. Git add, commit with descriptive message
5. Push to `origin/mambatest`

All operations are logged to `/tmp/branch_f_deploy.log`.

## Simulation Testing (WSL2+ROS side)

### Prerequisites

On the simulation machine, ensure:
- ROS Noetic sourced: `source /opt/ros/noetic/setup.bash`
- Catkin workspace built: `source /root/catkin_ws/devel/setup.bash`
- Conda environment active: `conda activate ros_py38`
- Loopback alias set: `ip addr add 192.168.233.250/32 dev lo`
- WSL2 mirrored networking enabled

### Pull Latest Checkpoints

```bash
cd /root/catkin_ws/src/vitfly-mambatest
git pull
```

### Run Tests

```bash
# Test F1-BC (Pure BC baseline) — uses best_model.pth
bash run_full_test.bash F BranchFModel bc

# Test F2-BC-Aug (BC + Data augmentation) — uses bc_aug_best_model.pth
bash run_full_test.bash F BranchFModel bc_aug

# Test F4-MultiSeq (BC + Multi-sequence, seq_len=8) — uses multiseq_best_model.pth
bash run_full_test.bash F BranchFModel multiseq
```

### Custom Test Parameters

```bash
# Override desired velocity (default: 5.0 m/s)
bash run_full_test.bash F BranchFModel bc 7.0

# Override sequence length for F4 (default: 8)
bash run_full_test.bash F BranchFModel multiseq 5.0 8
```

### Expected Test Behavior

Each test run will:
1. Reset the drone in Flightmare via ROS topics
2. Launch `evaluation_node.py` with a branch-specific tag
3. Run `run_competition.py` with the specified model
4. Execute 30 navigation attempts (~60s per scenario)
5. Save summary to `results/branch_F_<variant>_summary.yaml`

### Test Output Locations

| Artifact | Path |
|----------|------|
| Evaluation log | `/tmp/eval_F.log` |
| Competition log | `/tmp/comp_F.log` |
| Summary results | `envtest/ros/summary.yaml` |
| Saved results | `results/branch_F_<variant>_summary.yaml` |

## Results Collection

After testing all three variants, collect results for comparison:

```bash
# View individual summaries
cat results/branch_F_bc_summary.yaml
cat results/branch_F_bc_aug_summary.yaml
cat results/branch_F_multiseq_summary.yaml

# Compare crash counts across variants
echo "=== Crash Comparison ==="
for f in results/branch_F_*_summary.yaml; do
    variant=$(basename "$f" | sed 's/branch_F_//; s/_summary.yaml//')
    crashes=$(grep "crashes" "$f" | awk '{print $2}')
    echo "  ${variant}: ${crashes} crashes"
done
```

### Reporting Template

When reporting simulation results, include:

```
## Branch F Simulation Results

| Variant | Scenario | Speed | Crashes (10 runs) | Completion |
|---------|----------|-------|-------------------|------------|
| F1-BC | Spheres | 5 m/s | X/10 | X% |
| F1-BC | Spheres | 7 m/s | X/10 | X% |
| F1-BC | Trees | 5 m/s | X/10 | X% |
| F1-BC | Trees | 7 m/s | X/10 | X% |
| F2-BC-Aug | ... | ... | ... | ... |
| F4-MultiSeq | ... | ... | ... | ... |
```

## Troubleshooting

### Checkpoints Not Found on Pull

```bash
# Verify the deployment directory exists on GitHub
ls experiments/mamba_branches/optimized_training/branch_F/

# Run deployment again on the training machine
cd /root/vitfly && bash training/deploy_branch_f_checkpoints.sh
```

### Model Load Error in Simulation

```bash
# Verify checkpoint file integrity on simulation machine
python3 -c "
import torch
ckpt = torch.load('experiments/mamba_branches/optimized_training/branch_F/best_model.pth', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Config:', ckpt.get('config', 'N/A'))
"
```

### Simulation Fails to Start

Refer to the [VitFly Simulation Runbook](../../../.claude/skills/vitfly/SKILL.md) for:
- WSL2 network configuration
- Loopback route fixes
- ZMQ port cleanup (`wsl --shutdown`)
- OpenGL/Mesa configuration

## References

- [Experiment Design](EXPERIMENTS.md) — Full ablation study design
- [Model Architecture](README.md) — Branch F architecture details
- [Verification Report](VERIFICATION_REPORT.md) — Model verification results
- [VitFly Simulation Runbook](/.claude/skills/vitfly/SKILL.md) — WSL2 simulation setup
