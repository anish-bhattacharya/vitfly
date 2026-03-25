#!/usr/bin/env python3
"""分支 A: VMamba + LSTM 训练脚本"""

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
    p.add_argument('--data_dir', default='/root/catkin_ws/src/vitfly/training/datasets/data')
    p.add_argument('--exp_id', default='A_exp02')
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--lstm_hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', default='/root/catkin_ws/src/vitfly/experiments/mamba_branches/branch_A_vmamba_lstm/logs')
    p.add_argument('--device', default='cuda')
    p.add_argument('--short', type=int, default=0)
    p.add_argument('--warmup_epochs', type=int, default=0)
    args = p.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n使用设备：{device}", flush=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n加载数据...", flush=True)
    train_data, val_data, _, _ = dataloader(args.data_dir, val_split=args.val_split, seed=args.seed, short=args.short)
    traj_meta_train, traj_ims_train, _, desired_vels_train, curr_quats_train, _ = train_data
    traj_meta_val, traj_ims_val, _, desired_vels_val, curr_quats_val, _ = val_data
    print(f"训练集：{len(traj_ims_train)}, 验证集：{len(traj_ims_val)}", flush=True)
    
    train_loader = DataLoader(FlightmareDataset(traj_ims_train, traj_meta_train, desired_vels_train, curr_quats_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlightmareDataset(traj_ims_val, traj_meta_val, desired_vels_val, curr_quats_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("\n创建模型...", flush=True)
    model = VMambaLSTMNet(vmamba_config={'embed_dim': args.embed_dim, 'depth': args.depth, 'd_state': args.d_state, 'output_dim': 512}, lstm_hidden=args.lstm_hidden, lstm_layers=2, dropout=0.1).to(device)
    print(f"参数量：{model.get_parameter_count():,}", flush=True)
    
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    print(f"\n开始训练 ({args.epochs} epochs)...", flush=True)
    best_val, train_losses, val_losses = float('inf'), [], []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, opt, crit, device, epoch)
        val_loss = validate(model, val_loader, crit, device)
        sched.step()
        train_losses.append(train_loss); val_losses.append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {sched.get_last_lr()[0]:.6f}, Time: {time.time()-start:.1f}s", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss, 'config': args.__dict__}, f"{args.save_dir}/{args.exp_id}_best.pth")
            print(f"  ✓ 保存最佳模型", flush=True)
    np.save(f"{args.save_dir}/{args.exp_id}_train.npy", np.array(train_losses))
    np.save(f"{args.save_dir}/{args.exp_id}_val.npy", np.array(val_losses))
    print(f"\n训练完成！最佳 Val Loss: {best_val:.4f}", flush=True)

if __name__ == '__main__': main()
