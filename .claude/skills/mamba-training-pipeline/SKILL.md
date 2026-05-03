# Mamba Training Pipeline

Training pipeline for 5 Mamba-based obstacle avoidance models (Branch A-E) in the vitfly project.
Reference repo: https://github.com/anish-bhattacharya/vitfly (baseline)
Our fork: https://github.com/Liber1917/vitfly (branch: mambatest)

## Quick Start - Full Training

```bash
# Kill any existing training
pkill -9 -f train_mamba_optimized

# Start full training (5 branches × 100 epochs)
setsid python3 -u /root/vitfly/training/train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches A B C D E \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.0001 \
  --num_workers 2 \
  --clip_grad_norm 0.5 \
  --seed 42 \
  --val_split 0.2 \
  < /dev/null > /root/vitfly/training/logs/run_$(date +%m%d_%H%M).log 2>&1 &
```

**Critical**: Use `setsid` with `</dev/null >/dev/null 2>&1 &`. Do NOT use `nohup`, `screen`, or `bash -c` — these get killed by tool timeout.

## How to Launch Training (Reliably)

### Step 1: Prepare the start script
Write the command in `/root/vitfly/training/start_train.sh`:

```bash
#!/bin/bash
cd /root/vitfly/training
exec python3 -u train_mamba_optimized.py --data_dir /root/vitfly/training/datasets/data_full --branches A B C D E --epochs 100 --batch_size 64 --lr 0.0001 --num_workers 2 --clip_grad_norm 0.5 --seed 42 --val_split 0.2 > logs/run_$(date +%m%d_%H%M).log 2>&1
```

```bash
chmod +x /root/vitfly/training/start_train.sh
```

### Step 2: Launch
```bash
setsid /root/vitfly/training/start_train.sh </dev/null >/dev/null 2>&1 &
echo "PID: $!"
```

### Step 3: Verify
```bash
sleep 30 && ps aux | grep train_mamba
tail -f /root/vitfly/training/logs/run_*.log
```

## Dataset

- **Full dataset**: `/root/vitfly/training/datasets/data_full/` (580 trajectories, 3.4GB, ~110K PNG images)
- **Effective usable**: 253 trajectories, 42,156 images (327 folders auto-skipped due to row count mismatch)
- **Reference CSV header**: `timestamp,desired_vel,quat_1,quat_2,quat_3,quat_4,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,velcmd_x,velcmd_y,velcmd_z,ct_cmd,br_cmd_x,br_cmd_y,br_cmd_z,is_collide`

### Column Index Reference (data_full format)

| Columns | Content | Usage |
|---------|---------|-------|
| 0 | timestamp | - |
| 1 | desired_vel (SCALAR) | target normalization |
| 2-5 | quaternion (w,x,y,z) | model input (concat) |
| 6-8 | position (x,y,z) | - |
| **9-11** | **vel_x, vel_y, vel_z** | **model input** (`traj_meta[:, 10:13]`) |
| **12-14** | **velcmd_x, velcmd_y, velcmd_z** | **training target** (`traj_meta[:, 13:16]`) |
| 15-18 | ct_br_x..w | - |
| 19 | is_collide (0/1) | - |

```python
# Correct __getitem__ mapping (matching upstream):
velocity = torch.from_numpy(self.traj_meta[idx, 10:13]).float()
target   = torch.from_numpy(self.traj_meta[idx, 13:16]).float()
target   = target / self.traj_meta[idx, 2]        # normalize by desired_vel
```

## Commands

### Monitor Training
```bash
tail -f /root/vitfly/training/logs/run_*.log          # Live log stream
grep -c "Epoch.*Train Loss" /root/vitfly/training/logs/run_*.log  # Epoch count
grep -c NaN/Inf /root/vitfly/training/logs/run_*.log              # NaN check
ps aux | grep train_mamba                                        # Process alive?
grep "Training Branch" /root/vitfly/training/logs/run_*.log | tail -1  # Current branch
nvidia-smi                                                        # GPU status
```

### Kill Training
```bash
pkill -9 -f train_mamba_optimized
```

### Verify Checkpoints & Push to GitHub
```bash
/root/vitfly/training/verify_and_push.sh
```

### Check GPU & Disk
```bash
nvidia-smi                                                  # GPU usage, memory
df -h /                                                     # Disk usage
du -sh /root/.local/share/opencode/log/                      # OpenCode log size (can grow to 16GB+)
```

## 5 Branch Models

| Branch | Model Architecture | Parameters | Best Val Loss (100 epoch) |
|--------|-------------------|-----------|--------------------------|
| A | VMamba + LSTM | 0.68M | 0.0153 |
| B | MambaVision + SSM | 2.61M | 0.0192 |
| C | CNN + Mamba3 | 2.14M | 0.0178 |
| D | STH-Mamba | 2.76M | 0.0165 |
| E | DecisionMamba | 1.36M | **0.0142** |

## Training Config & Why

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | **0.0001** | 3e-4 causes NaN collapse at epoch 19 |
| `num_workers` | **2** | fastest 0-deadlock option; 4 causes deadlock |
| `batch_size` | **64** | fits 32GB GPU (~3% utilization only) |
| `clip_grad_norm` | **0.5** | tighter than default 1.0; prevents NaN |
| `epochs` | **100** | full convergence (cosine annealing reaches ~0 LR) |
| optimizer | AdamW (0.9, 0.95) | matches upstream |
| TF32 | enabled | ~2x speedup on RTX 5090 |
| warmup_epochs | 20 | linear warmup then cosine annealing |

### Timing (RTX 5090, num_workers=2)

- Per epoch: ~40 seconds
- 1 branch (100 epochs): ~67 minutes
- All 5 branches: ~5.5 hours

## Key Files

| Path | Purpose |
|------|---------|
| `training/train_mamba_optimized.py` | Main training script (507 lines) |
| `training/lazy_dataloading.py` | Lazy on-demand image loader |
| `training/start_train.sh` | Launch wrapper (setsid-ready) |
| `training/verify_and_push.sh` | Post-training verification + GitHub push |
| `training/logs/run_*.log` | Training logs |
| `experiments/mamba_branches/optimized_training/branch_*/` | Checkpoints per branch |

## Known Issues & Fixes

### NaN Loss at Epoch 19

**Symptom**: Train loss → 0.0000, Val loss → NaN after ~19 epochs. 42K+ NaN warnings.

**Root cause**: lr=3e-4 too high for this architecture. SSM layers diverge when warmup completes at peak LR.

**Fix**: Lower lr to **1e-4**. Tighten `clip_grad_norm` to **0.5**.

**Verification**: Run training. Check `grep -c NaN/Inf log` remains 0 after epoch 20.

### Process Dies on Launch

**Symptom**: Process starts then immediately disappears.

**Fix**: Use `setsid` with stdin/stdout/stderr redirected:
```bash
setsid python3 script.py </dev/null >log 2>&1 &
```

Not `nohup`, not `screen -dmS`, not `bash -c`.

### DataLoader Deadlock

**Symptom**: Process alive, zero progress, no output.

**Fix**: Use `num_workers=2` (or 0 if still failing).
num_workers=0: ~93s/epoch (safe but slow)
num_workers=2: ~40s/epoch (stable, verified)
num_workers=4: deadlock risk

### Disk Full

**Symptom**: Checkpoints fail to save, process dies silently.

**Check**: `df -h /`

**Known culprit**: `/root/.local/share/opencode/log/` (16GB+) — OpenCode session logs.

**Fix**: `rm -f /root/.local/share/opencode/log/old_*.log` or `df -h / && du -sh /root/.local/share/opencode/log/`.

### OMP_NUM_THREADS Warning

`libgomp: Invalid value for environment variable OMP_NUM_THREADS`

Harmless. Training continues normally. Can suppress with `OMP_NUM_THREADS=1`.

## GitHub Push

### Via HTTPS Token
```bash
git remote set-url origin "https://Liber1917:$(grep -oP 'ghp_\w+' ~/.config/gh/hosts.yml)@github.com/Liber1917/vitfly.git"
git push origin mambatest
```

### Via SSH
```bash
ssh-keygen -t ed25519 -f /tmp/vitfly_key -N ""
gh ssh-key add /tmp/vitfly_key.pub -t "vitfly-push"
git remote set-url origin git@github.com:Liber1917/vitfly.git
GIT_SSH_COMMAND="ssh -i /tmp/vitfly_key" git push origin mambatest
```

## History: Bug Timeline

1. **Target bug**: `target = velocity.clone()` → corrected to `traj_meta[:, 13:16]` (upstream convention)
2. **Memory leak**: Pre-loading 110K images = OOM → lazy dataloader (on-demand)
3. **NaN epoch 19**: lr=3e-4 too high → reduced to 1e-4 + clip=0.5
4. **Process handling**: nohup/screen killed by timeout → setsid </dev/null
5. **Column confusion**: `traj_meta[:, 2:5]` or `[:, 7:10]` → correct is `[:, 10:13]` (current velocity)
6. **Disk full**: OpenCode logs 16GB+ → periodic cleanup needed
