# Optimized Mamba Training Script

## Overview
This script provides optimized training for Mamba branches B-E with maximum GPU utilization.

## Key Features

### 1. Mixed Precision Training (FP16)
- Uses `torch.cuda.amp.GradScaler` and `autocast`
- Optimized gradient scaler settings for stability
- FP16 computation for faster training

### 2. Optimized DataLoader
- `num_workers=4` for parallel data loading
- `pin_memory=True` for faster GPU transfer
- `prefetch_factor=2` for data prefetching
- Proper 3D velocity extraction: `traj_meta[:, 2:5]`
- `drop_last=True` for stable gradient accumulation

### 3. GPU Memory Management
- Monitors GPU utilization with `nvidia-smi`
- Uses gradient accumulation for larger effective batch sizes
- Clears cache with `torch.cuda.empty_cache()` periodically
- Prints GPU memory usage every 50 batches

### 4. Model Support
- Branch B: `MambaVisionSSMNet` (~3.3M params)
- Branch C: `CNNMamba3Net` (~3.0M params)
- Branch D: `STHMambaNet`
- Branch E: `DecisionMambaNet`

### 5. Training Configuration
- Batch size: 32 (adjustable)
- Learning rate: 1e-4 with warmup
- Epochs: 100
- Gradient clipping: max_norm=1.0
- Gradient accumulation: adjustable steps
- Cosine annealing learning rate scheduler

## Usage

### Basic Usage
```bash
cd /root/vitfly/training
python train_mamba_optimized.py
```

### Train Specific Branches
```bash
python train_mamba_optimized.py --branches B C
```

### Custom Configuration
```bash
python train_mamba_optimized.py \
  --batch_size 64 \
  --grad_accum_steps 2 \
  --epochs 200 \
  --lr 2e-4 \
  --num_workers 8 \
  --save_dir /path/to/save
```

### Monitor GPU Usage
The script automatically monitors GPU memory usage and prints it every 50 batches:
```
GPU 0: 12456/24564 MB (50.7%) Epoch 1, Batch 50/1000
```

## Expected Performance

- **GPU Utilization**: > 80%
- **Training Time**: < 2 hours per branch (with 100 epochs)
- **Convergence**: Within 100 epochs
- **Checkpoints**: Saved every 25 epochs
- **Best Model**: Automatically saved when validation loss improves

## Output Structure

```
optimized_training/
├── branch_B/
│   ├── best_model.pth
│   ├── checkpoint_epoch_25.pth
│   ├── checkpoint_epoch_50.pth
│   ├── checkpoint_epoch_75.pth
│   ├── checkpoint_epoch_100.pth
│   ├── train_losses.npy
│   └── val_losses.npy
├── branch_C/
│   └── ...
├── branch_D/
│   └── ...
└── branch_E/
    └── ...
```

## Testing

Run the test script to verify everything works:
```bash
python test_mamba_optimized.py
```

## Requirements

- PyTorch 1.9+ with CUDA support
- NVIDIA GPU with sufficient memory (≥ 8GB recommended)
- Python 3.8+
- Required Python packages: torch, numpy, psutil

## Notes

1. The script automatically handles missing velocity data in metadata
2. Gradient accumulation allows for larger effective batch sizes without increasing memory usage
3. Mixed precision training provides ~2x speedup on compatible GPUs
4. The script clears GPU cache every 10 epochs to prevent memory fragmentation
5. All random seeds are set for reproducibility