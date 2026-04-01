# Vision Transformers (ViTs) for End-to-End Vision-Based Quadrotor Obstacle Avoidance (ICRA 2025)

[Project page](https://www.anishbhattacharya.com/research/vitfly)  &
[Paper](https://arxiv.org/abs/2405.10391)

This is the official repository for the paper "Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance" by Bhattacharya, et al. (2024) from GRASP, Penn.

We demonstrate that vision transformers (ViTs) can be used for end-to-end perception-based obstacle avoidance for quadrotors equipped with a depth camera. We train policies that predict linear velocity commands from depth images to avoid obstacles via behavior cloning from a privileged expert in a simple simulation environment, and show that ViT models combined with recurrence layers (LSTMs) outperform baseline methods based on other popular learning architectures.

## Project Structure

```
vitfly/
├── training/              # Model training scripts
│   ├── train_mamba_optimized.py  # Main training script for Mamba branches
│   ├── dataloading.py     # Dataset loading
│   └── config/            # Training configs
├── experiments/mamba_branches/  # Mamba branch model implementations
│   ├── branch_A_vmamba_lstm/    # VMamba + LSTM
│   ├── branch_B_mambavision_ssm/ # MambaVision + SSM
│   ├── branch_C_cnn_mamba3/     # CNN + Mamba3
│   ├── branch_D_sth_mamba/      # STH-Mamba
│   └── branch_E_decisionmamba/  # DecisionMamba
├── models/                # Original ViT-Fly models
├── flightmare/            # Quadrotor simulator
└── requirements.txt       # Dependencies
```

## Installation

#### Clone repository
```bash
cd ~/catkin_ws/src
git clone git@github.com:anish-bhattacharya/vitfly.git
cd vitfly
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### (Optional) Set up ROS/Flightmare
For simulation testing, see the original documentation. Additional details at https://github.com/uzh-rpg/agile_flight.

## Dataset Setup

Download `data.zip` (2.5GB, 580 trajectories) from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025):
```bash
mkdir -p training/datasets/data_full training/logs
unzip <path/to/data.zip> -d training/datasets/data_full
```

## Training

### Quick Start
```bash
cd training
python train_mamba_optimized.py --data_dir /root/vitfly/training/datasets/data_full
```

### Train Specific Branches
```bash
python train_mamba_optimized.py --branches B C D E
```

### Custom Configuration
```bash
python train_mamba_optimized.py \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.0001 \
  --num_workers 4 \
  --save_dir ./checkpoints
```

### Training Features
- Mixed Precision Training (FP16) with torch.cuda.amp
- Optimized DataLoader with parallel loading
- GPU memory monitoring
- Gradient accumulation for larger effective batch sizes
- Learning rate warmup and cosine annealing
- Checkpoint saving and validation

## Mamba Branch Results

| Branch | Model | Parameters | Best Val Loss |
|--------|-------|------------|---------------|
| A | VMamba+LSTM | ~3M | 0.00007 |
| B | MambaVision+SSM | ~2.6M | 0.000001 |
| C | CNN+Mamba3 | ~2.1M | 0.000001 |
| D | STH-Mamba | ~2.8M | 0.000001 |
| E | DecisionMamba | ~1.4M | 0.000007 |

All branches show convergence without overfitting when trained with sufficient data (200 trajectories).

## Key Bugs Fixed

### 1. Target Variable Bug (CRITICAL)
**Before**: `target = [desired_vels[idx]] * 3` (repeated scalar)
**After**: `target = velocity.clone()` (correct 3D velocity)

### 2. Empty Validation Set
- Fixed: sample-level split instead of trajectory-level for small datasets

### 3. Branch E Epochs
- Fixed: retrained with correct 100 epochs instead of default 10

## Testing (Simulation)

Download pretrained models from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025):
```bash
tar -xvf <path/to/pretrained_models.tar> -C models
bash launch_evaluation.bash 1 vision
```

## Citation

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgements

Simulation launching code and the versions of `flightmare` and `dodgedrone_simulation` are from the [ICRA 2022 DodgeDrone Competition code](https://github.com/uzh-rpg/agile_flight).