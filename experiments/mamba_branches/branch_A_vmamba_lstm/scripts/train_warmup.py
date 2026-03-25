#!/usr/bin/env python3
"""带 Warmup 的训练脚本"""
import os, sys, argparse, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

sys.path.insert(0, '/root/catkin_ws/src/vitfly')
sys.path.insert(0, '/root/catkin_ws/src/vitfly/training')
sys.path.insert(0, '/root/catkin_ws/src/vitfly/experiments/mamba_branches/branch_A_vmamba_lstm/models')

from dataloading import dataloader
from vmamba_lstm_model import VMambaLSTMNet

class FlightmareDataset(Dataset):
    def __init__(self, traj_ims, traj_meta, desired_vels, curr_quats):
        self.traj_ims, self.traj_meta = traj_ims, traj_meta
        self.desired_vels, self.curr_quats = np.asarray(desired_vels), curr_quats
    def __len__(self): return len(self.traj_ims)
    def __getitem__(self, idx):
        depth = torch.from_numpy(self.traj_ims[idx]).unsqueeze(0).float()
        target = torch.tensor([self.desired_vels[idx]] * 3, dtype=torch.float32)
        des_vel = torch.from_numpy(self.traj_meta[idx][:3]).float()
        quat = torch.from_numpy(self.curr_quats[idx]).float()
        return depth, des_vel, quat, target

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer, self.warmup_epochs, self.total_epochs, self.base_lr = optimizer, warmup_epochs, total_epochs, base_lr
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        return lr

def train_epoch(model, loader, opt, crit, device, epoch):
    model.train(); total_loss, n = 0, 0
    for i, (d, v, q, t) in enumerate(loader):
        d, v, q, t = d.to(device), v.to(device), q.to(device), t.to(device)
        opt.zero_grad()
        out, _ = model([d, v, q])
        loss = crit(out, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); n += 1
        if i % 50 == 0: print(f"  Epoch {epoch}, Batch {i}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)
    return total_loss / n

def validate(model, loader, crit, device):
    model.eval(); total_loss, n = 0, 0
    with torch.no_grad():
        for d, v, q, t in loader:
            d, v, q, t = d.to(device), v.to(device), q.to(device), t.to(device)
            out, _ = model([d, v, q])
            total_loss += crit(out, t).item(); n += 1
    return total_loss / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_id', default='A_exp03_warmup')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=4e-3)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--short', type=int, default=100)
    args = p.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备：{device}", flush=True)
    
    print("\n加载数据...", flush=True)
    train_data, val_data, _, _ = dataloader('/root/catkin_ws/src/vitfly/training/datasets/data', val_split=0.2, seed=42, short=args.short)
    traj_meta_train, traj_ims_train, _, desired_vels_train, curr_quats_train, _ = train_data
    traj_meta_val, traj_ims_val, _, desired_vels_val, curr_quats_val, _ = val_data
    print(f"训练集：{len(traj_ims_train)}, 验证集：{len(traj_ims_val)}", flush=True)
    
    train_loader = DataLoader(FlightmareDataset(traj_ims_train, traj_meta_train, desired_vels_train, curr_quats_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FlightmareDataset(traj_ims_val, traj_meta_val, desired_vels_val, curr_quats_val), batch_size=args.batch_size, shuffle=False)
    
    print("\n创建模型...", flush=True)
    model = VMambaLSTMNet(vmamba_config={'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512}, lstm_hidden=128, lstm_layers=2, dropout=0.1).to(device)
    print(f"参数量：{model.get_parameter_count():,}", flush=True)
    
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = WarmupCosineLR(opt, args.warmup_epochs, args.epochs, args.lr)
    
    print(f"\n开始训练 ({args.epochs} epochs, warmup={args.warmup_epochs})...", flush=True)
    best_val = float('inf')
    for epoch in range(args.epochs):
        start = time.time()
        lr = sched.step(epoch)
        train_loss = train_epoch(model, train_loader, opt, crit, device, epoch+1)
        val_loss = validate(model, val_loader, crit, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr:.6f}, Time: {time.time()-start:.1f}s", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'val_loss': val_loss}, f"/root/catkin_ws/src/vitfly/experiments/mamba_branches/branch_A_vmamba_lstm/logs/{args.exp_id}_best.pth")
            print(f"  ✓ 保存最佳模型 (Val={val_loss:.4f})", flush=True)
    print(f"\n训练完成！最佳 Val Loss: {best_val:.4f}", flush=True)

if __name__ == '__main__': main()
