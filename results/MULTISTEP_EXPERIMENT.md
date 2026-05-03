# Multi-Step Sequence Prediction: Experiment Design

## 1. Motivation

The upstream vitfly repository (anish-bhattacharya/vitfly) trains models on **full trajectory sequences**, computing loss over all time steps. Our current pipeline trains on **single frames** (`seq_len=1`), meaning the LSTM/SSM temporal heads process at most 1 step per forward pass and cannot leverage their recurrent capability.

This experiment measures the effect of multi-step training on prediction quality, output smoothness, and the accuracy-latency trade-off.

## 2. Hypothesis

**H0 (Null):** Multi-step training (seq_len > 1) does not improve simulation pass rate or velocity prediction accuracy compared to single-frame training.

**H1 (Alternative):** Multi-step training with 8-16 frame sequences reduces validation loss by >10% and produces smoother velocity commands compared to single-frame training, at the cost of ~2x training time per step.

## 3. Independent Variable

| Variable | Values | Unit |
|----------|--------|------|
| `--sequence_length` | 1, 4, 8, 16, 32 | frames |

Default: 1 (current baseline). All other hyperparameters stay fixed (see §5).

## 4. Dependent Variables

| Metric | Measurement | Priority |
|--------|------------|----------|
| Validation MSE loss | `criterion(output, target)` on val set | Primary |
| Output smoothness | Mean absolute diff between consecutive velocity predictions: `mean(|v[t+1] - v[t]|)` | Secondary |
| Training time per epoch | Wall clock for 1 epoch | Secondary |
| GPU memory usage | `nvidia-smi` peak memory | Secondary |

## 5. Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Branch B (MambaVisionSSM) | Most stable, known-working branch |
| Epochs | 1 per seq_len | Comparison metric, not convergence |
| Batch size | 32 | Lowered from 64 to fit seq_len=32 |
| Learning rate | 1e-4 | Prevents NaN (proven stable) |
| Optimizer | AdamW (β=0.9, 0.95) | Standard |
| Gradient clip | 0.5 | Prevents SSM explosion |
| Workers | 0 | Avoids DataLoader deadlock |
| Data | 100 trajectories (short=100) | Faster iteration for ablation |
| Seed | 42 | Reproducibility |
| GPU | RTX 5090 (32GB) | — |

## 6. Command to Run

```bash
cd /root/vitfly/training

# seq_len=1 (baseline)
setsid python3 -u train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches B --epochs 1 --batch_size 32 --lr 0.0001 \
  --num_workers 0 --clip_grad_norm 0.5 --seed 42 \
  --short 100 --sequence_length 1 \
  < /dev/null > logs/exp_seqlen1.log 2>&1 &

# seq_len=4
setsid python3 -u train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches B --epochs 1 --batch_size 32 --lr 0.0001 \
  --num_workers 0 --clip_grad_norm 0.5 --seed 42 \
  --short 100 --sequence_length 4 \
  < /dev/null > logs/exp_seqlen4.log 2>&1 &

# seq_len=8
setsid python3 -u train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches B --epochs 1 --batch_size 32 --lr 0.0001 \
  --num_workers 0 --clip_grad_norm 0.5 --seed 42 \
  --short 100 --sequence_length 8 \
  < /dev/null > logs/exp_seqlen8.log 2>&1 &

# seq_len=16
setsid python3 -u train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches B --epochs 1 --batch_size 32 --lr 0.0001 \
  --num_workers 0 --clip_grad_norm 0.5 --seed 42 \
  --short 100 --sequence_length 16 \
  < /dev/null > logs/exp_seqlen16.log 2>&1 &

# seq_len=32
setsid python3 -u train_mamba_optimized.py \
  --data_dir /root/vitfly/training/datasets/data_full \
  --branches B --epochs 1 --batch_size 32 --lr 0.0001 \
  --num_workers 0 --clip_grad_norm 0.5 --seed 42 \
  --short 100 --sequence_length 32 \
  < /dev/null > logs/exp_seqlen32.log 2>&1 &
```

## 7. Model Architecture Note

Current models expect input of shape `(B, ...)`. The sequence dataloader returns `(B, S, ...)` 
where S = sequence length. The training loop handles this by:

```python
# In train_epoch: when seq_len > 1, flatten batch and sequence dim
B, S = depth.shape[:2]
output, _ = model([depth.view(B*S, 1, 60, 90),
                    velocity.view(B*S, 3),
                    quat.view(B*S, 4)])
output = output.view(B, S, 3)  # (B, S, 3)
target = target.view(B, S, 3)  # (B, S, 3)
loss = criterion(output, target)  # MSE over all (B*S) elements
```

This processes each frame through the shared visual encoder independently,
then computes loss over all steps. The temporal head (LSTM/SSM) does NOT
receive the full sequence — each step is processed independently.
This is a conservative baseline; true sequential processing would require
modifying the temporal head to accept `(B, S, D)` inputs.

## 8. Expected Results

```
SEQ_LEN | Val Loss | Smoothness | Time/Epoch | Memory
--------|----------|------------|------------|--------
1       | 0.027    | 0.012      | ~30s       | 1.1GB
4       | 0.025    | 0.008      | ~35s       | 1.3GB
8       | 0.024    | 0.006      | ~45s       | 1.5GB
16      | 0.023    | 0.005      | ~65s       | 2.0GB
32      | 0.026    | 0.009      | ~110s      | 3.5GB
```

Key question: does val loss decrease with longer sequences (more data per sample)
or does it plateau/increase (temporal mismatches in the data)?

## 9. Analysis Protocol

### 9.1 After all runs complete

```bash
# Extract val loss from each run
for L in 1 4 8 16 32; do
    echo "SEQ_LEN=$L: $(grep 'Best validation loss' logs/exp_seqlen${L}.log)"
done

# Extract training time
for L in 1 4 8 16 32; do
    echo "SEQ_LEN=$L: $(grep 'Time:' logs/exp_seqlen${L}.log | tail -1)"
done

# Check for NaN/Inf
for L in 1 4 8 16 32; do
    echo "SEQ_LEN=$L NaN: $(grep -c 'NaN/Inf' logs/exp_seqlen${L}.log || echo 0)"
done
```

### 9.2 Interpretation

| Outcome | Interpretation | Recommendation |
|---------|---------------|----------------|
| seq_len=8/16 best | Temporal context helps | Use seq_len=8 or 16 for full training |
| seq_len=1 best | Multi-step adds noise | Keep single-frame pipeline |
| All similar | Model architecture dominates | Focus on encoder improvements |
| seq_len=32 best | Long context needed | Try seq_len=64+ |

### 9.3 Success Criteria

- **Minimum**: All seq_len values complete 1 epoch without NaN
- **Acceptable**: seq_len=8 or 16 matches or improves val loss vs baseline
- **Good**: seq_len=8 or 16 improves val loss by >5% AND output smoothness by >10%
- **Excellent**: seq_len=8 or 16 also shows qualitative improvement in simulation

## 10. Files & Artifacts

| Path | Content |
|------|---------|
| `logs/exp_seqlen{1,4,8,16,32}.log` | Raw training logs |
| `experiments/mamba_branches/optimized_training/branch_B/` | Checkpoints |
| `results/MULTISTEP_EXPERIMENT.md` | This experiment design |
| `training/lazy_dataloading.py` | SequenceFlightmareDataset + create_sequence_dataloader |

## 11. Dependencies

- `training/lazy_dataloading.py`: contains `SequenceFlightmareDataset` and `create_sequence_dataloader`
- `training/train_mamba_optimized.py`: supports `--sequence_length N` (default 1)
- `training/logs/` directory must exist

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sequence data too short after filtering | Fewer training samples | Use short=100 initially, scale up |
| Out of memory at seq_len=32 | Crash | Reduce batch_size to 16 if needed |
| NaN loss at long sequences | Wasted run | Use lr=1e-4 (proven stable) |
| DataLoader workers deadlock | Hang | num_workers=0 (verified safe) |
