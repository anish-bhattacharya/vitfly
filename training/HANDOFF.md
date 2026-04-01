# Training Handoff Documentation

## Quick Start

```bash
cd /root/vitfly/training

# Train single branch
python train_mamba_optimized.py --branches A --epochs 100

# Train multiple branches
python train_mamba_optimized.py --branches A B C D E --epochs 100

# Custom configuration
python train_mamba_optimized.py \
  --branches A \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0001 \
  --data_dir /root/vitfly/training/datasets/data_full \
  --val_split 0.2
```

## Project Structure

```
training/
├── train_mamba_optimized.py   # Main training script (use this)
├── train.py                   # Original training script
├── dataloading.py             # Data loading utilities
├── config/                    # Training configs
├── datasets/
│   └── data_full/            # Dataset (580 trajectories)
└── experiments/mamba_branches/
    └── optimized_training/   # Output directory
```

## Available Branches

| Branch | Model | Description |
|--------|-------|-------------|
| A | VMamba+LSTM | VMamba encoder + LSTM |
| B | MambaVision+SSM | MambaVision + SSM |
| C | CNN+Mamba3 | CNN + Mamba3 |
| D | STH-Mamba | STH-Mamba |
| E | DecisionMamba | Decision Mamba |

## Output Files

After training, checkpoints are saved to:
```
experiments/mamba_branches/optimized_training/branch_{X}/
├── best_model.pth           # Best model (lowest val loss)
├── checkpoint_epoch_{N}.pth # Epoch checkpoints
├── train_losses.npy         # Training loss history
└── val_losses.npy           # Validation loss history
```

## Known Issues

### 1. Empty Validation Set
If using small dataset with trajectory-level split, validation set may be empty.
- **Fix**: Use sample-level split by ensuring `val_split * len(traj_folders) >= 1`

### 2. Target Variable Bug (FIXED)
Original code used repeated scalar `[desired_vels[idx]] * 3` instead of proper 3D velocity.
- **Status**: Fixed in current version - uses `velocity.clone()`

### 3. Data Mismatch Warnings
Some trajectories show "Number of images and telemetry still do not match" - these are skipped automatically.

## Dataset

Download from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025):
- `data.zip` (2.5GB, 580 trajectories)

Extract to: `training/datasets/data_full/`

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--branches` | A B C D E | Branches to train |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--val_split` | 0.2 | Validation split |
| `--num_workers` | 4 | DataLoader workers |
| `--data_dir` | ./datasets/data | Data directory |
| `--save_dir` | ../experiments/mamba_branches/optimized_training | Output dir |

## Monitoring

```bash
# View training logs
tensorboard --logdir training/logs

# Check GPU usage
nvidia-smi
```

## Verification

Run quick verification:
```bash
python train_mamba_optimized.py --branches A --epochs 1 --short 10 --val_split 0.2
```

Expected output:
```
Training samples: X
Validation samples: Y
Epoch   1/1 | Train Loss: X.XXXX | Val Loss: X.XXXX | LR: 0.000100 | Time: Xs
```

If Val Loss shows `inf`, the validation set may be empty - check data loading output.