#!/usr/bin/env python3
"""
Optimized training script for Mamba branches with maximum GPU utilization.

Supports: A, B, C, D, E, Bplus, Fusion, Essm, F

Key Features:
1. Mixed Precision Training (FP16) with torch.cuda.amp
2. Optimized DataLoader with parallel loading
3. GPU memory monitoring and management
4. Gradient accumulation for larger effective batch sizes
5. Learning rate warmup and cosine annealing
6. Checkpoint saving and validation
7. YAML config file support

Expected Performance:
- GPU utilization > 80%
- Training time < 2 hours per branch
- Convergence within 100 epochs
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.amp import autocast, GradScaler
import subprocess
import psutil
import yaml

# Add paths to import models
sys.path.insert(0, '/root/vitfly')
sys.path.insert(0, '/root/vitfly/training')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_A_vmamba_lstm/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_B_mambavision_ssm/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_C_cnn_mamba3/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_D_sth_mamba/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_Bplus_mambavision_mamba3/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/mambafusion/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/essm/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_F_lightweight_mamba3/models')

# Import models
try:
    from vmamba_lstm_model import VMambaLSTMNet, create_vmamba_lstm_model
    from mambavision_ssm_model import MambaVisionSSMNet, create_mambavision_ssm_model
    from cnn_mamba3_model import CNNMamba3Net, create_cnn_mamba3_model
    from sth_mamba_model import STHMambaNet, create_sth_mamba_model
    from decision_mamba_model import DecisionMambaNet, create_decision_mamba_model
    from bplus_model import create_bplus_model
    from mambafusion_model import create_mambafusion_model
    from essm_model import create_essm_model
    from branch_f_model import create_branch_f_model
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import some models: {e}")
    MODELS_AVAILABLE = False

# Import dataloaders
from dataloading import dataloader
from lazy_dataloading import create_lazy_dataloader, create_sequence_dataloader


class OptimizedFlightmareDataset(Dataset):
    """Optimized dataset with proper 3D velocity extraction."""
    
    def __init__(self, traj_ims, traj_meta, desired_vels, curr_quats):
        """
        Args:
            traj_ims: List of depth images
            traj_meta: Metadata array with shape (N, M)
            desired_vels: Desired velocities (scalar)
            curr_quats: Current quaternions
        """
        self.traj_ims = traj_ims
        self.traj_meta = traj_meta
        self.desired_vels = np.asarray(desired_vels)
        self.curr_quats = curr_quats
        
    def __len__(self):
        return len(self.traj_ims)
    
    def __getitem__(self, idx):
        depth = torch.from_numpy(self.traj_ims[idx]).unsqueeze(0).float()

        # Extract 3D velocity from traj_meta[:, 2:5] (current velocity - MODEL INPUT)
        if self.traj_meta.shape[1] >= 5:
            velocity = torch.from_numpy(self.traj_meta[idx, 2:5]).float()
        else:
            velocity = torch.zeros(3, dtype=torch.float32)

        quat = torch.from_numpy(self.curr_quats[idx]).float()

        # TARGET: Use expert velocity COMMAND from columns 13:16
        # Reference: vitfly/training/train.py line 185 → self.train_velcmd = self.train_meta[:, range(13, 16)]
        # NOT velocity (cols 2:5) which is the model INPUT - using that causes mode collapse
        target = torch.from_numpy(self.traj_meta[idx, 13:16]).float()
        target = target / (torch.norm(target) + 1e-6)

        return depth, velocity, quat, target


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            memory_info = []
            for line in lines:
                used, total = map(int, line.split(','))
                memory_info.append({
                    'used_mb': used,
                    'total_mb': total,
                    'usage_percent': (used / total) * 100
                })
            return memory_info
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
    
    # Fallback to torch.cuda
    if torch.cuda.is_available():
        return [{
            'used_mb': torch.cuda.memory_allocated() / 1024**2,
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
            'usage_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        }]
    
    return None


def print_gpu_memory_usage(epoch, batch_idx, total_batches):
    """Print current GPU memory usage."""
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        for i, info in enumerate(gpu_info):
            print(f"  GPU {i}: {info['used_mb']:.0f}/{info['total_mb']:.0f} MB ({info['usage_percent']:.1f}%) "
                  f"Epoch {epoch}, Batch {batch_idx}/{total_batches}")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(branch_name, config, device, args=None):
    """Create model based on branch name."""
    if branch_name == 'A':
        model = create_vmamba_lstm_model(config)
    elif branch_name == 'B':
        model = create_mambavision_ssm_model(config)
    elif branch_name == 'C':
        model = create_cnn_mamba3_model(config)
    elif branch_name == 'D':
        model = create_sth_mamba_model(config)
    elif branch_name == 'E':
        model = create_decision_mamba_model(config)
    elif branch_name == 'Bplus':
        model = create_bplus_model(config)
    elif branch_name == 'Fusion':
        model = create_mambafusion_model(config)
    elif branch_name == 'Essm':
        model = create_essm_model(config)
    elif branch_name == 'F':
        model = create_branch_f_model(config)
    else:
        raise ValueError(f"Unknown branch: {branch_name}")
    
    model = model.to(device)
    
    if args and args.compile and branch_name not in ('A', 'Bplus'):
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print(f"  torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    
    return model


def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, 
                grad_accum_steps=1, clip_grad_norm=1.0, seq_len=1):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (depth, velocity, quat, target) in enumerate(loader):
        depth = depth.to(device, non_blocking=True)
        velocity = velocity.to(device, non_blocking=True)
        quat = quat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        if torch.isnan(depth).any() or torch.isnan(velocity).any() or torch.isnan(quat).any() or torch.isnan(target).any():
            print(f"  Warning: NaN detected in batch {batch_idx}, skipping")
            continue
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            if seq_len > 1:
                B, S = depth.shape[:2]
                depth_f = depth.view(B * S, 1, depth.shape[-2], depth.shape[-1])
                vel_f = velocity.reshape(B * S, -1)
                quat_f = quat.reshape(B * S, -1)
                output, _ = model([depth_f, vel_f, quat_f])
                output = output.reshape(B, S, -1)
                target_f = target.reshape(B, S, -1)
            else:
                output, _ = model([depth, velocity, quat])
                target_f = target
            loss = criterion(output, target_f)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                continue
            
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if batch_idx % 50 == 0:
                print_gpu_memory_usage(epoch, batch_idx, len(loader))
                print(f"    Batch {batch_idx}/{len(loader)}, Loss: {loss.item() * grad_accum_steps:.4f}")
        
        total_loss += loss.item() * grad_accum_steps
        total_samples += depth.size(0)
    
    if total_samples % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, seq_len=1):
    """Validate model performance."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for depth, velocity, quat, target in loader:
            depth = depth.to(device, non_blocking=True)
            velocity = velocity.to(device, non_blocking=True)
            quat = quat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                if seq_len > 1:
                    B, S = depth.shape[:2]
                    depth_f = depth.view(B * S, 1, depth.shape[-2], depth.shape[-1])
                    vel_f = velocity.reshape(B * S, -1)
                    quat_f = quat.reshape(B * S, -1)
                    output, _ = model([depth_f, vel_f, quat_f])
                    output = output.reshape(B, S, -1)
                    target_f = target.reshape(B, S, -1)
                else:
                    output, _ = model([depth, velocity, quat])
                    target_f = target
            
            total_loss += criterion(output, target_f).item()
    
    return total_loss / len(loader)


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if warmup_epochs >= total_epochs:
            return 1.0
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_branch(branch_name, args, train_loader, val_loader, device, model_config=None):
    """Train a single branch."""
    print(f"\n{'='*60}")
    print(f"Training Branch {branch_name}")
    print(f"{'='*60}")
    
    if model_config:
        config = model_config
        if isinstance(config, dict) and 'architecture' in config:
            del config['architecture']
    else:
        config = {
        'mambavision_config': {
            'in_channels': 1,
            'stem_dim': 48,
            'stage_dims': (64, 128, 192),
            'depths': (2, 2, 2),
            'd_state': 12,
            'dropout': 0.1,
            'output_dim': 512
        },
        'ssm_d_state': 16,
        'ssm_hidden': 256,
        'ssm_layers': 2,
        'dropout': 0.1
    }
    
    # Create model
    model = create_model(branch_name, config, device, args)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    # Gradient scaler for mixed precision with optimized settings
    scaler = GradScaler('cuda', init_scale=2.**16, growth_interval=2000)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create save directory for this branch
    branch_save_dir = os.path.join(args.save_dir, f"branch_{branch_name}")
    os.makedirs(branch_save_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch,
            grad_accum_steps=args.grad_accum_steps, clip_grad_norm=args.clip_grad_norm,
            seq_len=args.sequence_length
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, seq_len=args.sequence_length)
        
        # Update learning rate
        scheduler.step()
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            checkpoint_path = os.path.join(branch_save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(branch_save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            print(f"  New best model! Val Loss: {val_loss:.4f}")
        
        # Clear GPU cache periodically
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final losses
    np.save(os.path.join(branch_save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(branch_save_dir, "val_losses.npy"), np.array(val_losses))
    
    print(f"\nBranch {branch_name} training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Optimized training for Mamba branches B-E")
    
    # Data arguments
    parser.add_argument('--data_dir', default='/root/vitfly/training/datasets/data',
                       help='Directory containing training data')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--short', type=int, default=0,
                       help='Use only first N trajectories (0=all)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile (mode=reduce-overhead) for non-A branches')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Data loader prefetch factor')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable data augmentation (horizontal flip + noise)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda or cpu)')
    
    # Branch selection
    parser.add_argument('--branches', nargs='+', default=['A', 'B', 'C', 'D', 'E'],
                        help='Branches to train (A, B, C, D, E, Bplus, Fusion, Essm, F)')
    parser.add_argument('--sequence_length', type=int, default=1,
                        help='Sequence length for multi-step temporal training (1=standard single-frame)')
    # Config file
    parser.add_argument('--config', default=None,
                       help='Path to YAML config file (overrides hardcoded defaults)')
    # Output arguments
    parser.add_argument('--save_dir', default='/root/vitfly/experiments/mamba_branches/optimized_training',
                       help='Directory to save checkpoints and logs')
    
    args = parser.parse_args()
    
    # Load YAML config if provided (overrides hardcoded defaults)
    model_config_override = None
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                yaml_cfg = yaml.safe_load(f)
            # Override training args from YAML
            if 'training' in yaml_cfg:
                tr = yaml_cfg['training']
                args.epochs = tr.get('epochs', args.epochs)
                args.batch_size = tr.get('batch_size', args.batch_size)
                args.lr = tr.get('learning_rate', args.lr)
                args.grad_accum_steps = tr.get('grad_accum_steps', args.grad_accum_steps)
                args.clip_grad_norm = tr.get('clip_grad_norm', args.clip_grad_norm)
                args.augment = tr.get('augment', args.augment)
                args.sequence_length = tr.get('sequence_length', args.sequence_length)
                if tr.get('scheduler', {}).get('warmup_epochs') is not None:
                    args.warmup_epochs = tr['scheduler']['warmup_epochs']
            # Store model config section for create_model
            model_config_override = yaml_cfg.get('model')
            # Override save_dir if specified
            if yaml_cfg.get('save_dir'):
                args.save_dir = yaml_cfg['save_dir']
            if yaml_cfg.get('data_dir'):
                args.data_dir = yaml_cfg['data_dir']
            if yaml_cfg.get('num_workers'):
                args.num_workers = yaml_cfg['num_workers']
            print(f"Loaded config from: {args.config}")
        else:
            print(f"Warning: Config file not found: {args.config}")
    
    # Set seed
    set_seed(args.seed)
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Check if models are available
    if not MODELS_AVAILABLE:
        print("Error: Could not import all models. Please check model paths.")
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.sequence_length > 1:
        print(f"\nLoading data with SEQUENCE dataloader (seq_len={args.sequence_length})...")
        train_loader, val_loader, stats = create_sequence_dataloader(
            data_dir=args.data_dir, seq_len=args.sequence_length,
            val_split=args.val_split, batch_size=args.batch_size,
            num_workers=args.num_workers, short=args.short, seed=args.seed,
            pin_memory=True,
            augment=args.augment
        )
        print(f"Training sequences: {stats['num_train']}")
        print(f"Validation sequences: {stats['num_val']}")
    else:
        print("\nLoading data with lazy dataloader...")
        train_loader, val_loader, stats = create_lazy_dataloader(
            data_dir=args.data_dir,
            val_split=args.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            short=args.short,
            seed=args.seed,
            pin_memory=True,
            augment=args.augment
        )
        print(f"Training samples: {stats['num_train_samples']}")
        print(f"Validation samples: {stats['num_val_samples']}")
        print(f"Trajectories loaded: {stats['num_trajectories']}")
    
    if val_loader is None:
        print("Warning: No validation data available")
        from torch.utils.data import TensorDataset
        empty_dataset = TensorDataset(torch.zeros(0, 1, 60, 90), torch.zeros(0, 3), torch.zeros(0, 4), torch.zeros(0, 3))
        val_loader = DataLoader(
            empty_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    print(f"\nDataLoader configuration:")
    print(f"  Workers: {args.num_workers}")
    print(f"  Pin memory: True")
    
    # Train each branch
    results = {}
    for branch in args.branches:
        if branch not in ['A', 'B', 'C', 'D', 'E', 'Bplus', 'Fusion', 'Essm', 'F']:
            print(f"Warning: Unknown branch '{branch}', skipping...")
            continue
        
        # Clear GPU cache before training each branch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train branch
        best_val_loss = train_branch(branch, args, train_loader, val_loader, device, model_config=model_config_override)
        results[branch] = best_val_loss
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for branch, loss in results.items():
        print(f"Branch {branch}: Best validation loss = {loss:.4f}")
    
    print(f"\nAll branches trained successfully!")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()