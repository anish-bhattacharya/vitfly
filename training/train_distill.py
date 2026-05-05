#!/usr/bin/env python3
"""
Distillation Training: ViT+LSTM Teacher → Mamba Student

Cross-architecture knowledge distillation for quadrotor obstacle avoidance.
Multi-stage alignment based on MOHAWK (NeurIPS 2024), CAB (2025), X-Distill (ICLR 2026).

Loss Design:
  L = α · L_feat + β · L_distill + γ · L_gt

  L_feat    = MSE(student_encoder_feat, aligned_teacher_feat)   — MOHAWK Stage 2
  L_distill = MSE(student_output, teacher_output)                — MOHAWK Stage 3
  L_gt      = MSE(student_output, ground_truth)                  — standard BC

Teacher: ViT+LSTM (LSTMNetVIT, best upstream model, 7m/s real flight)
Students: 6 Mamba-based branches (A, B, B+, C, D, E)

Usage:
  # ── Ablation experiments (all on Branch B, vary loss weights) ──
  # C0: Pure BC baseline (no distillation)
  python train_distill.py --branch B --epochs 100 --alpha 0 --beta 0 --gamma 1
  
  # C1: Feature alignment only
  python train_distill.py --branch B --epochs 100 --alpha 1 --beta 0 --gamma 1
  
  # C2: Output distillation only (tests MOHAWK "naive KD fails" claim)
  python train_distill.py --branch B --epochs 100 --alpha 0 --beta 1 --gamma 1
  
  # C3: Full multi-stage distillation (recommended default)
  python train_distill.py --branch B --epochs 100 --alpha 1 --beta 1 --gamma 1
  
  # ── Cross-architecture comparison (fixed loss, vary branch) ──
  python train_distill.py --all-branches --epochs 100 --alpha 1 --beta 1 --gamma 1
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.amp import autocast, GradScaler

# ─── Path Setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/root/vitfly')
sys.path.insert(0, '/root/vitfly/training')
sys.path.insert(0, '/root/vitfly/models')

BRANCH_PATHS = {
    'A': '/root/vitfly/experiments/mamba_branches/branch_A_vmamba_lstm/models',
    'B': '/root/vitfly/experiments/mamba_branches/branch_B_mambavision_ssm/models',
    'Bplus': '/root/vitfly/experiments/mamba_branches/branch_Bplus_mambavision_mamba3/models',
    'C': '/root/vitfly/experiments/mamba_branches/branch_C_cnn_mamba3/models',
    'D': '/root/vitfly/experiments/mamba_branches/branch_D_sth_mamba/models',
    'E': '/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models',
}
for path in BRANCH_PATHS.values():
    sys.path.insert(0, path)

# ─── Imports ──────────────────────────────────────────────────────────────────
# Teacher: Use TeacherVITLSTM from shared models/model.py (checkpoint-compatible).
# The upstream ViTLSTM_model.pth has lstm input_size=517 (scalar desired_vel),
# while the original LSTMNetVIT uses 519 (3D velocity) — can't load the checkpoint.
from model import TeacherVITLSTM

from vmamba_lstm_model import VMambaLSTMNet, create_vmamba_lstm_model
from mambavision_ssm_model import MambaVisionSSMNet, create_mambavision_ssm_model
from cnn_mamba3_model import CNNMamba3Net, create_cnn_mamba3_model
from sth_mamba_model import STHMambaNet, create_sth_mamba_model
from decision_mamba_model import DecisionMambaNet, create_decision_mamba_model
from bplus_model import BPlusModel, create_bplus_model

from lazy_dataloading import create_lazy_dataloader, create_sequence_dataloader


# ─── Configuration ────────────────────────────────────────────────────────────

# Which module attribute on each branch model produces visual encoder features
VISUAL_ENCODER_ATTR = {
    'A': 'vmamba',          # VMambaLSTMNet.vmamba → 512-dim
    'B': 'mambavision',     # MambaVisionSSMNet.mambavision → 512-dim
    'Bplus': 'mambavision', # BPlusModel.mambavision → 512-dim
    'C': 'cnn',             # CNNMamba3Net.cnn (CNNEncoder) → 512-dim
    'D': 'spatial_encoder', # STHMambaNet.spatial_encoder → 256-dim
    'E': 'cnn_encoder',     # DecisionMambaNet.cnn_encoder → 256-dim
}

# Visual feature dimension for each branch
VISUAL_FEATURE_DIM = {
    'A': 512, 'B': 512, 'Bplus': 512,
    'C': 512, 'D': 256, 'E': 256,
}

TEACHER_FEATURE_DIM = 512  # LSTMNetVIT.decoder output

# Model factory mapping
BRANCH_CREATORS = {
    'A': lambda cfg: create_vmamba_lstm_model(cfg.get('vmamba_config', {})),
    'B': lambda cfg: create_mambavision_ssm_model(cfg),
    'Bplus': lambda cfg: create_bplus_model(cfg),
    'C': lambda cfg: create_cnn_mamba3_model(cfg),
    'D': lambda cfg: create_sth_mamba_model(cfg),
    'E': lambda cfg: create_decision_mamba_model(cfg),
}

# Default hyperparameters per branch (from BC training in train_mamba_optimized.py)
# CRITICAL: must match exactly — these configs determine model architecture.
# Branch C's cnn_config is intentionally omitted: BC pipeline passes None too,
# letting the model class use its own default (stage_dims=(32,64,128,256)).
# Branch A and E use {} in BC code; their creators fall back to model defaults.
DEFAULT_CONFIG = {
    'mambavision_config': {
        'in_channels': 1, 'stem_dim': 48,
        'stage_dims': (64, 128, 192), 'depths': (2, 2, 2),
        'd_state': 12, 'dropout': 0.1, 'output_dim': 512
    },
    'ssm_d_state': 16, 'ssm_hidden': 256, 'ssm_layers': 2, 'dropout': 0.1,
}


# ─── Feature Projection ───────────────────────────────────────────────────────

class FeatureProjector(nn.Module):
    """Learnable linear projection for mismatched feature dimensions.
    
    When student visual feature dim ≠ teacher feature dim (512),
    project student features to match teacher space.
    """
    def __init__(self, in_dim, out_dim=TEACHER_FEATURE_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.proj(x)


# ─── Hook Manager ─────────────────────────────────────────────────────────────

class FeatureHook:
    """Register forward hook to capture intermediate features."""
    def __init__(self):
        self.features = {}
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def register_teacher(self, teacher_model, teacher_branch=None):
        """Register hook on teacher's visual encoder.
        
        For ViT+LSTM teacher: hooks decoder (512-dim).
        For Mamba teacher: hooks the branch's visual encoder module.
        """
        if teacher_branch is not None and teacher_branch.lower() != 'teacher':
            attr = VISUAL_ENCODER_ATTR.get(teacher_branch)
            if attr is None:
                raise ValueError(f"No visual encoder attr for branch {teacher_branch}")
            encoder_module = getattr(teacher_model, attr)
        else:
            encoder_module = teacher_model.decoder
        handle = encoder_module.register_forward_hook(
            self._hook_fn('teacher_visual')
        )
        return handle
    
    def register_student(self, student_model, branch):
        """Register hook on student's visual encoder module."""
        attr = VISUAL_ENCODER_ATTR.get(branch)
        if attr is None:
            raise ValueError(f"No visual encoder attr defined for branch {branch}")
        encoder_module = getattr(student_model, attr)
        handle = encoder_module.register_forward_hook(
            self._hook_fn('student_visual')
        )
        return handle
    
    def get_teacher_feat(self):
        """Get captured teacher visual features (B, 512)."""
        return self.features.get('teacher_visual')
    
    def get_student_feat(self):
        """Get captured student visual features."""
        return self.features.get('student_visual')
    
    def clear(self):
        self.features = {}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """Dataset wrapper that provides consistent (depth, velocity, quat, target) tuples.
    
    Compatible with LazyFlightmareDataset format — used for both teacher and student.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        # base dataset returns (depth, velocity, quat, target)
        return self.base[idx]


# ─── Loss Functions ───────────────────────────────────────────────────────────

def compute_distillation_loss(
    student_output,        # (B, 3) student velocity prediction
    teacher_output,        # (B, 3) teacher velocity prediction (frozen)
    student_feat,          # (B, D_s) student visual features
    teacher_feat,          # (B, 512) teacher visual features (detached)
    ground_truth,          # (B, 3) ground truth velocity command
    projector,             # FeatureProjector or None
    alpha=1.0,             # feature alignment weight
    beta=1.0,              # output distillation weight
    gamma=1.0,             # GT supervision weight
):
    """Compute combined distillation loss.
    
    Args are already on the correct device.
    Returns dict of individual losses and weighted total.
    """
    # Feature alignment loss (MOHAWK Stage 2)
    if student_feat is not None and teacher_feat is not None:
        s_feat = student_feat.float()
        t_feat = teacher_feat.float()
        # Project student features if dims don't match
        if projector is not None:
            s_feat = projector(s_feat)
        loss_feat = nn.functional.mse_loss(s_feat, t_feat)
    else:
        loss_feat = torch.tensor(0.0, device=student_output.device)
    
    # Output distillation loss (MOHAWK Stage 3)
    loss_distill = nn.functional.mse_loss(student_output, teacher_output.detach())
    
    # Ground truth supervision
    loss_gt = nn.functional.mse_loss(student_output, ground_truth)
    
    # Weighted total
    loss_total = alpha * loss_feat + beta * loss_distill + gamma * loss_gt
    
    return {
        'loss': loss_total,
        'loss_feat': loss_feat,
        'loss_distill': loss_distill,
        'loss_gt': loss_gt,
    }


# ─── Training Functions ───────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            mem_info = []
            for line in lines:
                used, total = map(int, line.split(','))
                mem_info.append({'used_mb': used, 'total_mb': total,
                                 'usage_percent': (used / total) * 100})
            return mem_info
    except Exception:
        pass
    if torch.cuda.is_available():
        return [{
            'used_mb': torch.cuda.memory_allocated() / 1024**2,
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
            'usage_percent': (torch.cuda.memory_allocated() / 
                              torch.cuda.get_device_properties(0).total_memory) * 100
        }]
    return None


def load_teacher_for_branch(teacher_branch, device, teacher_checkpoint):
    """Load a Mamba branch model as teacher (for born-again distillation)."""
    from mambavision_ssm_model import MambaVisionSSMNet, create_mambavision_ssm_model
    from bplus_model import BPlusModel, create_bplus_model
    from cnn_mamba3_model import CNNMamba3Net, create_cnn_mamba3_model
    from sth_mamba_model import STHMambaNet, create_sth_mamba_model
    from decision_mamba_model import DecisionMambaNet, create_decision_mamba_model
    from vmamba_lstm_model import VMambaLSTMNet, create_vmamba_lstm_model

    config = DEFAULT_CONFIG.copy()
    creator = BRANCH_CREATORS.get(teacher_branch)
    if creator is None:
        raise ValueError(f"Unknown teacher branch: {teacher_branch}")
    teacher = creator(config).to(device).eval()
    print(f"  Teacher branch: {teacher_branch}")

    if os.path.exists(teacher_checkpoint):
        ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=True)
        sd = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        teacher.load_state_dict(sd, strict=False)
        print(f"  Loaded teacher from: {teacher_checkpoint}")
    else:
        print(f"  WARNING: Teacher checkpoint not found at {teacher_checkpoint}")

    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def load_teacher(device, teacher_checkpoint='/root/vitfly/models/ViTLSTM_model.pth',
                 teacher_branch=None):
    """Load frozen teacher model.
    
    Args:
        teacher_branch: None for ViT+LSTM, or branch name for born-again (B+, E, etc.)
    """
    if teacher_branch is not None and teacher_branch.lower() != 'teacher':
        return load_teacher_for_branch(teacher_branch, device, teacher_checkpoint)
    
    print(f"\n{'='*60}")
    print("Loading Teacher: ViT+LSTM (TeacherVITLSTM, ckpt-compatible)")
    print(f"{'='*60}")
    
    teacher = TeacherVITLSTM().to(device)
    teacher.eval()
    
    if os.path.exists(teacher_checkpoint):
        ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=True)
        if any(k.startswith('encoder_blocks.') for k in ckpt.keys()):
            teacher.load_state_dict(ckpt, strict=False)
        elif 'model_state_dict' in ckpt:
            teacher.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'state_dict' in ckpt:
            teacher.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"  Loaded teacher from: {teacher_checkpoint}")
    else:
        print(f"  WARNING: Teacher checkpoint not found at {teacher_checkpoint}")
    
    for param in teacher.parameters():
        param.requires_grad = False
    
    teacher_param_count = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher parameters: {teacher_param_count:,} ({teacher_param_count/1e6:.2f}M)")
    print(f"  Teacher frozen: True")
    
    return teacher


def create_student(branch, device, args):
    """Create a student model for the given branch.
    
    Architecture must match what train_mamba_optimized.py produces.
    Uses the same config dict and same factory functions.
    """
    config = DEFAULT_CONFIG.copy()
    
    creator = BRANCH_CREATORS.get(branch)
    if creator is None:
        raise ValueError(f"Unknown branch: {branch}")
    
    student = creator(config).to(device)
    
    # Load BC checkpoint as initialization (if requested)
    if args.init_from_bc:
        bc_path = os.path.join(args.save_dir, f"branch_{branch}", "best_model.pth")
        if os.path.exists(bc_path):
            bc_ckpt = torch.load(bc_path, map_location=device, weights_only=True)
            sd = bc_ckpt.get('model_state_dict', bc_ckpt)
            # Handle torch.compile prefix
            if any(k.startswith('_orig_mod.') for k in sd.keys()):
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            try:
                student.load_state_dict(sd, strict=False)
                bc_val_loss = bc_ckpt.get('val_loss', '?')
                bc_epoch = bc_ckpt.get('epoch', '?')
                print(f"  Loaded BC checkpoint: {bc_path}")
                print(f"    BC epoch={bc_epoch}, val_loss={bc_val_loss}")
            except Exception as e:
                print(f"  WARNING: BC checkpoint load failed: {e}")
                print(f"  Falling back to random init.")
        else:
            print(f"  WARNING: BC checkpoint not found at {bc_path}")
            print(f"  Training from scratch (random init).")
    
    # Optionally compile (reduce-overhead mode for non-A branches)
    if args.compile and branch not in ('A', 'Bplus'):
        try:
            student = torch.compile(student, mode="reduce-overhead", dynamic=True)
            print(f"  torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    
    return student


def setup_feature_projection(branch, device):
    """Create feature projector if student feature dim ≠ teacher feature dim."""
    student_dim = VISUAL_FEATURE_DIM.get(branch, 512)
    if student_dim != TEACHER_FEATURE_DIM:
        projector = FeatureProjector(student_dim, TEACHER_FEATURE_DIM).to(device)
        print(f"  Feature projector: {student_dim} → {TEACHER_FEATURE_DIM} (Linear)")
        return projector
    return None


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing LR scheduler with linear warmup."""
    def lr_lambda(epoch):
        if warmup_epochs >= total_epochs:
            return 1.0
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_distill_epoch(
    teacher, student, projector, loader, optimizer, scaler, device, epoch,
    hook, alpha=1.0, beta=1.0, gamma=1.0, grad_accum_steps=1, clip_grad_norm=0.5,
    seq_len=1, T=1.0
):
    """Train one epoch with distillation."""
    teacher.eval()  # always eval (frozen)
    student.train()
    
    total_loss = 0.0
    total_loss_feat = 0.0
    total_loss_distill = 0.0
    total_loss_gt = 0.0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (depth, velocity, quat, target) in enumerate(loader):
        depth = depth.to(device, non_blocking=True)
        velocity = velocity.to(device, non_blocking=True)
        quat = quat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Check for NaN
        if (torch.isnan(depth).any() or torch.isnan(velocity).any() or 
            torch.isnan(quat).any() or torch.isnan(target).any()):
            print(f"  Warning: NaN in batch {batch_idx}, skipping")
            continue
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Reshape for seq_len > 1: (B, S, ...) -> (B*S, ...)
            if seq_len > 1:
                B, S = depth.shape[:2]
                depth_f = depth.reshape(B * S, 1, depth.shape[-2], depth.shape[-1])
                vel_f = velocity.reshape(B * S, -1)
                quat_f = quat.reshape(B * S, -1)
                target_f = target.reshape(B, S, -1)
            else:
                depth_f, vel_f, quat_f = depth, velocity, quat
                target_f = target
            
            # ── Teacher forward (frozen, no grad) ──
            with torch.no_grad():
                hook.clear()
                teacher_out_tmp = teacher([depth_f, vel_f, quat_f])[0]
                teacher_feat = hook.get_teacher_feat()
            
            # ── Student forward ──
            hook.clear()
            student_out, _ = student([depth_f, vel_f, quat_f])
            student_feat = hook.get_student_feat()
            
            # Reshape for seq mode
            if seq_len > 1:
                student_out = student_out.reshape(B, S, -1)
                teacher_out = teacher_out_tmp.reshape(B, S, -1)
            else:
                teacher_out = teacher_out_tmp
            
            # ── Compute distillation loss ──
            if seq_len > 1:
                # seq mode: use last timestep for loss computation
                student_out_last = student_out[:, -1, :]
                teacher_out_last = teacher_out[:, -1, :]
                
                # For seq mode, compute losses only on last timestep output
                loss_dict = compute_distillation_loss(
                    student_out_last, teacher_out_last,
                    student_feat, teacher_feat,
                    target_f[:, -1, :],
                    projector, alpha, beta, gamma
                )
            else:
                loss_dict = compute_distillation_loss(
                    student_out, teacher_out,
                    student_feat, teacher_feat,
                    target, projector, alpha, beta, gamma
                )
            
            loss = loss_dict['loss'] / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate
        total_loss += loss_dict['loss'].item()
        total_loss_feat += loss_dict['loss_feat'].item()
        total_loss_distill += loss_dict['loss_distill'].item()
        total_loss_gt += loss_dict['loss_gt'].item()
        total_samples += depth.size(0)
    
    # Handle remainder
    if total_samples % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'loss_feat': total_loss_feat / n_batches,
        'loss_distill': total_loss_distill / n_batches,
        'loss_gt': total_loss_gt / n_batches,
    }


@torch.no_grad()
def validate_distill(teacher, student, projector, loader, device, hook, seq_len=1):
    """Validate student model with multi-metric tracking.
    
    Metrics:
      val_loss_gt:        MSE vs ground truth — monitor for divergence/collapse
      val_distill_gap:    MSE vs teacher output — tracks convergence toward teacher
      val_feat_align:     MSE of (projected) student vs teacher features
      val_action_mag:     Mean |student_output| — is student matching teacher's action scale?
      val_action_var:     Variance of student output within batch — action diversity proxy
    """
    teacher.eval()
    student.eval()
    
    total_loss_gt = 0.0
    total_distill_gap = 0.0
    total_feat_align = 0.0
    total_action_mag = 0.0
    total_action_var = 0.0
    total_action_max_vx = 0.0
    total_action_max_vy = 0.0
    total_action_max_vz = 0.0
    n_batches = 0
    
    for depth, velocity, quat, target in loader:
        depth = depth.to(device, non_blocking=True)
        velocity = velocity.to(device, non_blocking=True)
        quat = quat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        n_batches += 1
        
        # Reshape for seq_len > 1: (B, S, ...) -> (B*S, ...)
        if seq_len > 1:
            B, S = depth.shape[:2]
            depth_f = depth.reshape(B * S, 1, depth.shape[-2], depth.shape[-1])
            vel_f = velocity.reshape(B * S, -1)
            quat_f = quat.reshape(B * S, -1)
            target_f = target.reshape(B, S, -1)
        else:
            depth_f, vel_f, quat_f = depth, velocity, quat
            target_f = target
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # ── Teacher forward ──
            hook.clear()
            teacher([depth_f, vel_f, quat_f])
            teacher_out = teacher([depth_f, vel_f, quat_f])[0]
            teacher_feat = hook.get_teacher_feat()
            
            # ── Student forward ──
            hook.clear()
            if seq_len > 1:
                student_out, _ = student([depth_f, vel_f, quat_f])
                student_out = student_out.reshape(B, S, -1)
                teacher_out = teacher_out.reshape(B, S, -1)
                target_f = target.reshape(B, S, -1)
                # Use last timestep for single-frame metrics
                student_last = student_out[:, -1, :]
                target_last = target_f[:, -1, :]
            else:
                student_last, _ = student([depth, velocity, quat])
                target_last = target
            
            student_feat = hook.get_student_feat()
            
            # ── Metric 1: GT loss ──
            loss_gt = nn.functional.mse_loss(student_last, target_last)
            
            # ── Metric 2: Teacher agreement (distill gap) ──
            teacher_last = teacher_out[:, -1, :] if seq_len > 1 else teacher_out
            loss_distill = nn.functional.mse_loss(student_last, teacher_last)
            
            # ── Metric 3: Feature alignment ──
            if student_feat is not None and teacher_feat is not None:
                s_feat = student_feat.float()
                t_feat = teacher_feat.float()
                if projector is not None:
                    s_feat = projector(s_feat)
                feat_loss = nn.functional.mse_loss(s_feat, t_feat)
            else:
                feat_loss = torch.tensor(0.0)
            
            # ── Metric 4: Action magnitude (mean absolute output) ──
            action_mag = student_last.abs().mean()
            
            # ── Metric 5: Action variance within batch (diversity proxy) ──
            action_var = student_last.var(dim=0).mean()
        
        total_loss_gt += loss_gt.item()
        total_distill_gap += loss_distill.item()
        total_feat_align += feat_loss.item()
        total_action_mag += action_mag.item()
        total_action_var += action_var.item()
    
    n = n_batches
    return {
        'val_loss_gt': total_loss_gt / n,
        'val_distill_gap': total_distill_gap / n,
        'val_feat_align': total_feat_align / n,
        'val_action_mag': total_action_mag / n,
        'val_action_var': total_action_var / n,
    }


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train_distillation(branch, args, train_loader, val_loader, device):
    """Run distillation training for a single branch."""
    print(f"\n{'='*60}")
    print(f"Distillation Training: Branch {branch}")
    print(f"  Teacher: ViT+LSTM (frozen)")
    print(f"  Student: {branch}")
    print(f"  Loss weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"{'='*60}")
    
    # Create teacher
    teacher = load_teacher(device, args.teacher_ckpt, teacher_branch=args.teacher_branch)
    
    # Create student
    student = create_student(branch, device, args)
    student_param_count = sum(p.numel() for p in student.parameters())
    print(f"  Student ({branch}) parameters: {student_param_count:,} ({student_param_count/1e6:.2f}M)")
    
    # Feature projector (if needed)
    projector = setup_feature_projection(branch, device)
    
    # Feature hooks
    hook = FeatureHook()
    hook.register_teacher(teacher, teacher_branch=args.teacher_branch)
    
    # Register student hook — handle compiled model
    try:
        hook.register_student(student, branch)
    except Exception as e:
        print(f"  Warning: Student hook registration failed: {e}")
        print(f"  This may happen with torch.compile. Feature alignment disabled.")
        # Feature alignment won't work, but distillation still runs
        if projector is not None:
            projector = None
    
    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler('cuda', init_scale=2.**16, growth_interval=2000)
    
    # Training loop
    # Best model tracks combined: GT_loss + 0.5 * distill_gap
    # This prefers models that stay close to GT while converging toward teacher
    best_val_score = float('inf')
    best_val_gt = float('inf')
    best_val_distill = float('inf')
    train_history = []
    val_history = []
    
    # Save directory
    branch_save_dir = os.path.join(args.save_dir, f"branch_{branch}")
    os.makedirs(branch_save_dir, exist_ok=True)
    
    print(f"\nStarting distillation for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}, Grad accum: {args.grad_accum_steps}")
    print(f"  Sequence length: {args.sequence_length}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_distill_epoch(
            teacher, student, projector, train_loader, optimizer, scaler, device, epoch,
            hook, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            grad_accum_steps=args.grad_accum_steps, clip_grad_norm=args.clip_grad_norm,
            seq_len=args.sequence_length
        )
        
        # Validate
        val_metrics = validate_distill(
            teacher, student, projector, val_loader, device, hook,
            seq_len=args.sequence_length
        )
        
        # LR schedule
        scheduler.step()
        
        # Record
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Print
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Tr:{train_metrics['loss']:.4f} "
              f"(feat:{train_metrics['loss_feat']:.4f} "
              f"dist:{train_metrics['loss_distill']:.4f} "
              f"gt:{train_metrics['loss_gt']:.4f}) | "
              f"Val gt:{val_metrics['val_loss_gt']:.4f} "
              f"dist:{val_metrics['val_distill_gap']:.4f} "
              f"feat:{val_metrics['val_feat_align']:.4f} | "
              f"mag:{val_metrics['val_action_mag']:.3f} "
              f"var:{val_metrics['val_action_var']:.4f} | "
              f"LR:{current_lr:.6f} | "
              f"{epoch_time:.0f}s")
        
        # Save checkpoint
        if epoch % 25 == 0:
            ckpt_path = os.path.join(branch_save_dir, f"distill_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'branch': branch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'loss_weights': {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma},
                'config': args,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")
        
        # Save best (combined score: GT_loss + 0.5 * distill_gap)
        val_score = val_metrics['val_loss_gt'] + 0.5 * val_metrics['val_distill_gap']
        if val_score < best_val_score:
            best_val_score = val_score
            best_val_gt = val_metrics['val_loss_gt']
            best_val_distill = val_metrics['val_distill_gap']
            best_path = os.path.join(branch_save_dir, "distill_best_model.pth")
            torch.save({
                'epoch': epoch,
                'branch': branch,
                'model_state_dict': student.state_dict(),
                'val_loss_gt': best_val_gt,
                'val_distill_gap': best_val_distill,
                'val_score': best_val_score,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'loss_weights': {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma},
            }, best_path)
            print(f"  ★ Best model! score={best_val_score:.4f} "
                  f"(gt={best_val_gt:.4f} distill={best_val_distill:.4f})")
        
        # Periodic GPU cache clear
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save training history
    np.save(os.path.join(branch_save_dir, "distill_train_history.npy"), train_history)
    np.save(os.path.join(branch_save_dir, "distill_val_history.npy"), val_history)
    
    # Save summary report
    summary = {
        'branch': branch,
        'epochs': args.epochs,
        'best_val_score': best_val_score,
        'best_val_loss_gt': best_val_gt,
        'best_val_distill_gap': best_val_distill,
        'loss_weights': {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma},
        'student_params': student_param_count,
        'first_epoch_loss': train_history[0]['loss'] if train_history else None,
        'last_epoch_loss': train_history[-1]['loss'] if train_history else None,
        'teacher_checkpoint': args.teacher_ckpt,
        'init_from_bc': args.init_from_bc,
    }
    with open(os.path.join(branch_save_dir, "distill_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBranch {branch} distillation completed!")
    print(f"  Best val_score: {best_val_score:.4f} (gt={best_val_gt:.4f} distill={best_val_distill:.4f})")
    
    return best_val_score


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-architecture Knowledge Distillation: ViT+LSTM → Mamba"
    )
    
    # Distillation parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Feature alignment loss weight (MOHAWK Stage 2)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Output distillation loss weight (MOHAWK Stage 3)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Ground truth supervision weight')
    parser.add_argument('--teacher-ckpt', type=str,
                        default='/root/vitfly/models/ViTLSTM_model.pth',
                        help='Path to teacher model checkpoint')
    parser.add_argument('--teacher-branch', type=str, default=None,
                        choices=[None, 'A', 'B', 'Bplus', 'C', 'D', 'E', 'teacher'],
                        help='Branch name for born-again teacher (None=ViT+LSTM)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softening teacher logits')
    
    # Branch
    parser.add_argument('--branch', type=str, default='B',
                        choices=['A', 'B', 'Bplus', 'C', 'D', 'E'],
                        help='Student branch to train')
    parser.add_argument('--all-branches', action='store_true',
                        help='Train all 6 branches sequentially')
    
    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/root/vitfly/training/datasets/data_full',
                        help='Training data directory')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--short', type=int, default=0,
                        help='Use only N trajectories (0=all, for testing)')
    parser.add_argument('--sequence-length', type=int, default=1,
                        help='Sequence length for temporal training (1=standard)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-grad-norm', type=float, default=0.5)
    parser.add_argument('--init-from-bc', action='store_true', default=False,
                        help='Load BC checkpoint as student initialization (from save_dir/branch_{X}/best_model.pth)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile')
    
    # System
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (2 for stability)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', type=str,
                        default='/root/vitfly/experiments/mamba_branches/optimized_training',
                        help='Save directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
    
    # Validate teacher checkpoint exists
    if not os.path.exists(args.teacher_ckpt):
        print(f"\n  ⚠ WARNING: Teacher checkpoint not found at {args.teacher_ckpt}")
        print(f"  Distillation will use randomly initialized teacher.")
        print(f"  Place teacher checkpoint at this path for proper distillation.\n")
    
    # Select branches
    branches = [args.branch]
    if args.all_branches:
        branches = ['A', 'B', 'Bplus', 'C', 'D', 'E']
    
    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data (shared across branches for fair comparison)
    print(f"\nLoading data from: {args.data_dir}")
    if args.sequence_length > 1:
        train_loader, val_loader, stats = create_sequence_dataloader(
            data_dir=args.data_dir, seq_len=args.sequence_length,
            val_split=args.val_split, batch_size=args.batch_size,
            num_workers=args.num_workers, short=args.short, seed=args.seed,
            pin_memory=True
        )
        print(f"  Training sequences: {stats['num_train']}")
        print(f"  Validation sequences: {stats['num_val']}")
    else:
        train_loader, val_loader, stats = create_lazy_dataloader(
            data_dir=args.data_dir, val_split=args.val_split,
            batch_size=args.batch_size, num_workers=args.num_workers,
            short=args.short, seed=args.seed, pin_memory=True
        )
        print(f"  Training samples: {stats['num_train_samples']}")
        print(f"  Validation samples: {stats['num_val_samples']}")
    
    if val_loader is None:
        print("  Warning: No validation data. Creating empty validator.")
        from torch.utils.data import TensorDataset
        empty_ds = TensorDataset(
            torch.zeros(0, 1, 60, 90), torch.zeros(0, 3),
            torch.zeros(0, 4), torch.zeros(0, 3)
        )
        val_loader = DataLoader(empty_ds, batch_size=args.batch_size, num_workers=0)
    
    # Train each branch
    results = {}
    for branch in branches:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n{'#'*60}")
        print(f"# Starting distillation for Branch {branch}")
        print(f"{'#'*60}")
        
        val_score = train_distillation(branch, args, train_loader, val_loader, device)
        results[branch] = val_score
    
    # Summary
    print(f"\n{'='*60}")
    print("Distillation Summary")
    print(f"{'='*60}")
    # BC baselines from actual checkpoints
    bc_baselines = {'A': 0.0161, 'B': 0.0205, 'Bplus': 0.0231, 
                    'C': 0.0221, 'D': 0.0173, 'E': 0.0186}
    for branch, score in sorted(results.items()):
        baseline = bc_baselines.get(branch, '?')
        if isinstance(baseline, float):
            delta = score - baseline
            print(f"  Branch {branch}: val_score={score:.4f}  "
                  f"(BC gt_loss={baseline:.4f}, Δ={delta:+.4f})")
        else:
            print(f"  Branch {branch}: val_score={score:.4f}")
    print(f"\nCheckpoints saved to: {args.save_dir}")
    print(f"Loss weights: α={args.alpha} (feat), β={args.beta} (distill), γ={args.gamma} (GT)")


if __name__ == '__main__':
    main()
