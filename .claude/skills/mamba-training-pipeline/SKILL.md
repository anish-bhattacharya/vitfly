# Mamba Training Pipeline

General-purpose training pipeline for Mamba-based models. Covers training techniques, acceleration, distributed execution, and debugging for small-to-medium models (<10M params) on consumer GPUs.

**Vitfly project context** (for reference): 
- [Official ViTFly](https://github.com/anish-bhattacharya/vitfly) — ViT+LSTM quadrotor obstacle avoidance (ICRA 2025)
- [Our fork](https://github.com/Liber1917/vitfly) — Mamba variants + cross-architecture distillation

---

## 1. Reliable Process Launching

### The Golden Rule: Always Use `setsid`

When launching long-running training from non-interactive shells (CI, agent tools), processes get killed when the parent shell exits:

```bash
# ✅ CORRECT: detaches from parent shell completely
setsid python3 -u train_script.py --args... </dev/null >log.log 2>&1 &

# ❌ WRONG: killed when parent exits
nohup python3 train_script.py ... &
screen -dmS train python3 train_script.py ... &
bash -c "python3 train_script.py ..." &
```

**Mechanism**: `setsid` creates a new session leader, detaching the child from the launching shell's process group. The child survives terminal/SSH/tool timeouts.

### Best Practice: Wrapper Script

```bash
# start_train.sh
#!/bin/bash
cd /project/training
exec python3 -u train.py --epochs 100 > logs/run_$(date +%m%d_%H%M).log 2>&1
```

```bash
chmod +x start_train.sh
setsid ./start_train.sh </dev/null >/dev/null 2>&1 &
```

---

## 2. GPU Utilization Optimization

### 2.1 Problem: Models Under 10M Params = GPU Starvation

Small models (<10M params) use <5% of modern GPU compute. The bottleneck is **CPU-side data loading**, not GPU arithmetic.

**Signs of GPU starvation**:
- `nvidia-smi` shows <10% GPU utilization
- GPU memory usage is low (<20% of VRAM)
- Training loop shows long gaps between batches

### 2.2 DataLoader Tuning

| `num_workers` | Throughput | Risk |
|---|---|---|
| 0 | 1× (baseline) | None — safest, for sequence data |
| 2 | 2-3× faster | ✅ Sweet spot for most setups |
| 4 | ~2.5× | ❌ Deadlock risk on some systems |
| 8+ | Marginal gain | ❌ Diminishing returns |

```python
# Rule of thumb:
if seq_len > 1: num_workers = 0  # sequence dataloader often incompatible with multiprocessing
else:            num_workers = 2  # best tradeoff for throughput
```

### 2.3 Mixed Precision (FP16/BF16)

**Always enable AMP** for 2-3× speedup on Tensor Core GPUs (RTX 30xx+, A100+, H100):

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Warning**: Some SSM layers (selective scan) have numerical edge cases in FP16. Monitor for NaN/Inf after switching to mixed precision.

### 2.4 TF32 Tensor Cores (RTX 30xx+, A100+)

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Provides ~2× matmul speedup with negligible accuracy loss. **Works with FP32 training code** — transparent acceleration.

### 2.5 torch.compile

```python
model = torch.compile(model, mode="reduce-overhead", dynamic=True)
```

- `mode="reduce-overhead"`: Best for small models (<10M). Compilation overhead is amortized over many forward passes.
- `dynamic=True`: Handle variable batch sizes.
- **Caveat**: Compiled model state_dict keys have `_orig_mod.` prefix when saved. Strip before loading:
  ```python
  sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
  ```
- Not compatible with all architectures (e.g., VMamba's SS2D may fail). Test per-model.

### 2.6 NVIDIA MPS (Multi-Process Service) for Multi-Experiment Parallelism

When training multiple small models (<5M params each) on a single GPU, GPU utilization is typically <5% per process. MPS allows concurrent execution.

**When to use**: 2+ independent training runs with models that individually underutilize the GPU.

```bash
# Start MPS daemon
nvidia-cuda-mps-control -d

# Launch N parallel processes, each with 100/N % GPU share
for i in $(seq 0 3); do
  PCT=$((100 / 4))
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT \
  python3 train_C.py &
  python3 train_D.py &
  python3 train_E.py &
  python3 train_Bplus.py &
done

# Stop MPS daemon
echo quit | nvidia-cuda-mps-control
```

**Typical gains** (4 parallel runs, RTX 5090):
- GPU utilization: 3% → 12% (4×)
- Runtime: 4.4h sequential → 1.5h parallel (2.9×)

**Limitations**: 
- Only helps when individual processes are GPU-starved (<10% util)
- Each process gets ~5% VRAM for our small models — total VRAM still well within 32GB
- Not needed when single process already saturates GPU

---

## 3. Training Configuration Principles

### 3.1 Learning Rate

For SSM-based models (Mamba, S4, etc.), the learning rate is **lower than typical CNNs/Transformers**:

```python
# ✅ Safe for SSMs
lr = 1e-4       # AdamW, SSM models <10M params
clip = 0.5      # Gradient clipping threshold

# ❌ Too high — causes NaN collapse around epoch 15-20
lr = 3e-4       # SSM layers diverge at peak LR after warmup
```

**Why SSMs need lower LR**: The selective scan mechanism has recurrent dynamics that amplify gradient magnitudes, especially during the parameterized state transition (A matrix). High LR causes the selective scan to diverge numerically.

### 3.2 Warmup + Cosine Annealing Schedule

```python
def lr_schedule(epoch, warmup=20, total=100):
    if epoch < warmup:
        return (epoch + 1) / warmup  # linear warmup
    progress = (epoch - warmup) / (total - warmup)
    return 0.5 * (1 + cos(π * progress))  # cosine decay
```

- Warmup period: ~20% of total epochs
- This prevents early divergence from random initialization
- Cosine decay provides smooth convergence without plateau

### 3.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value=0.5)
```

**Rule of thumb**:
- SSM-only models: clip=0.5 (tighter)
- CNN/Attention hybrids: clip=1.0
- If NaN occurs: halve both LR and clip value

### 3.4 Batch Size for Small Models

Small models (<10M params) have low arithmetic intensity. Large batch sizes don't improve throughput:

```python
batch_size = 32   # sweet spot for models <5M params
batch_size = 64   # moderate throughput gain, more VRAM
```

Beyond 64, throughput saturates (data loading becomes the bottleneck).

---

## 4. Cross-Architecture Knowledge Distillation

### 4.1 When Distillation Helps

Cross-architecture distillation (Teacher → Student) is beneficial when:
- Teacher has **higher capacity** but is **too slow for deployment**
- Student has **complementary inductive biases** (linear SSM vs quadratic attention)
- Teacher was trained on **different data distribution** (higher speed, more data)

### 4.2 Multi-Stage Loss Design

Naive output-only distillation fails for cross-architecture transfer (proved by MOHAWK, NeurIPS 2024). A three-component loss is required:

```python
# Feature alignment (MOHAWK Stage 2 analog)
L_feat = MSE(student_visual_features, teacher_visual_features)

# Output distillation (MOHAWK Stage 3 analog)
L_distill = MSE(student_output, teacher_output)

# Ground truth supervision (prevents teacher bias collapse)
L_gt = MSE(student_output, ground_truth)

# Combined
L = α * L_feat + β * L_distill + γ * L_gt
```

**Default weights**: α=β=γ=1.0. These should be grid-searched per architecture pair.

### 4.3 Feature Dimension Handling

When student and teacher have different visual feature dimensions, learn a linear projection:

```python
if student_feat_dim != teacher_feat_dim:
    projector = nn.Linear(student_feat_dim, teacher_feat_dim)
    student_feat = projector(student_feat)
```

### 4.4 Interpreting Distillation Results

Key metrics to track during distillation:

| Metric | What It Measures | Good Signal |
|--------|-----------------|-------------|
| `val_loss_gt` | MSE vs ground truth | Not diverging from BC baseline |
| `val_distill_gap` | MSE vs teacher output | Decreasing toward teacher |
| `val_feat_align` | Visual feature similarity | Decreasing → student learning teacher's features |
| `action_mag` | Average |output| | Stable within 0.3-0.4 |
| `action_max_vz` | Max vertical velocity | <1.5 to avoid panic responses |

**Best model selection**: Use a combined score:
```python
val_score = val_loss_gt + 0.5 * val_distill_gap
```
This rewards both GT fidelity AND teacher alignment.

### 4.5 Born-Again Iterative Distillation

When a distilled student outperforms its teacher (validated in simulation), that student can become the teacher for a second round:

```
Round 1: ViT+LSTM teacher → Mamba student (cross-architecture)
Round 2: Mamba student teacher → Another Mamba student (same-architecture)
```

Same-architecture distillation typically shows 3-4× faster convergence (distill gap drops faster) because the teacher and student share the same inductive biases.

---

## 5. Multi-Step Sequence Training

For models with recurrent components (LSTM, SSM temporal heads), training on sequences >1 step allows the temporal mechanism to learn dynamics:

```python
# Data format: reshape batch to (B*S, C, H, W) for encoder
# Reshape output back to (B, S, 3), compute loss on last timestep
```

```bash
# Single-step (default)
python3 train.py --sequence_length 1

# Multi-step (16-frame sequences)
python3 train.py --sequence_length 16 --num_workers 0
```

**Note**: Sequence mode requires `num_workers=0` for data loader stability.

---

## 6. Training Progress Monitoring & Time Estimation

### 6.0 Multi-Process Status Check

When running multiple parallel experiments (MPS, born-again, grid search), get a complete picture at a glance:

```bash
# 1. Count all training processes
ps aux | grep -E "train_mamba|train_distill" | grep -v grep | wc -l

# 2. Check each experiment's current epoch and per-epoch timing
echo "=== MPS seq training ==="
grep "Epoch.*Time" logs/bc_seq4_*.log 2>/dev/null | tail -1
grep "Epoch.*Time" logs/bc_seq8_*.log 2>/dev/null | tail -1
grep "Epoch.*Time" logs/bc_seq16_*.log 2>/dev/null | tail -1

echo "=== Born-again ==="
grep "Epoch   " logs/bornagain_*.log 2>/dev/null | tail -1

echo "=== Grid search ==="
for f in logs/grid_*.log; do
  last=$(grep "Epoch   " "$f" 2>/dev/null | tail -1)
  echo "  $(basename $f): $last"
done

# 3. GPU status
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
```

### 6.1 Time Estimation Formula

When experiments run at different epochs/seq_lens, use this systematic estimation approach:

```
Remaining time = (total_epochs - current_epoch) × per_epoch_time

For parallel experiments, wall clock time = max(remaining time of all experiments)
The slowest sequential bottleneck determines completion, NOT the average.
```

**Example** (3 parallel MPS experiments + independent processes):
```
MPS seq=4:  31/100, 83s/ep  → 69 × 83 = 95min  ← SLOWEST = wall clock
MPS seq=8:  32/100, 79s/ep  → 68 × 79 = 90min
MPS seq=16: 35/100, 75s/ep  → 65 × 75 = 81min
Born-again:  9/50,  44s/ep  → 41 × 44 = 30min  ← finishes earlier
Grid ×4:     9/50,  42s/ep  → 41 × 42 = 29min  ← finishes earlier
──────────────────────────────────────────────────
Wall clock ≈ 95min (bounded by seq=4, which has most batches/ep)
```

**Key insight**: With MPS, multiple experiments share GPU but the epoch time per experiment remains similar to running alone (bottleneck is CPU data loading, not GPU compute). So time estimation is additive of the slowest, not multiplicative.

### 6.2 Detecting Missing/Dead Processes

When launching many parallel experiments (e.g., 4 grid variants in a bash loop), the launch script may be interrupted:

```bash
# Check: do all expected log files exist?
ls -la logs/grid_*.log           # Count actual vs expected
tail -1 logs/grid_*.log          # Any stuck at init?

# Re-launch only missing ones
for ALPHA in 0.5 1.0; do
  for BETA in 0.5 1.0; do
    TAG="a${ALPHA}_b${BETA}"
    LOG="logs/grid_E_${TAG}_*.log"
    if ! ls $LOG 2>/dev/null | grep -q .; then
      setsid python3 train_script.py --alpha $ALPHA --beta $BETA ...
    fi
  done
done
```

### 6.3 Critical: Isolate Experiment Output Directories

**NEVER let multiple experiments write to the same output directory.** The `--save-dir` parameter must be unique per experiment type.

#### What Goes Wrong

```
# ❌ WRONG: All experiments share the same save directory
born-again:  --save-dir optimized_training/branch_E   # saves distill_best_model.pth
grid ×4:     --save-dir optimized_training/branch_E   # OVERWRITES born-again's checkpoint!
```

The last experiment to finish overwrites all previous checkpoints and summaries, silently destroying hours of training.

#### Correct Directory Structure

```
optimized_training/
├── branch_E/               # Original BC + first distill (ViT+LSTM teacher)
├── bornagain_Bplus2E/      # Born-again isolate
├── grid_E_alpha_beta/      # Grid search isolate
└── seq_E_16/               # Sequence training isolate
```

#### Launch Pattern

```bash
# Each experiment type gets its own --save-dir
setsid python3 train_distill.py --branch E --teacher-branch Bplus \
  --save-dir experiments/mamba_branches/optimized_training/bornagain_Bplus2E \
  ... &

setsid python3 train_distill.py --branch E --alpha 0.5 --beta 0.5 \
  --save-dir experiments/mamba_branches/optimized_training/grid_E_alpha_beta \
  ... &
```

**Best practice**: Align `--save-dir` with the experiment name, e.g., `{experiment_type}_{description}/`. This prevents ambiguity and enables clean checkpoint management.

### 6.3b Checkpoint Distribution: Copy, Don't Symlink

When training and evaluation run on different machines (e.g., GPU training server vs ROS simulation workstation), the filesystems are not shared.

#### What Goes Wrong

```
# ❌ WRONG: Symlink in git
git add branch_E/distill_best_model.pth  # symlink to /root/vitfly/experiments/...
# Remote clone has no /root/vitfly/ directory → BROKEN LINK

# ❌ WRONG: Relative symlink
ln -s ../seq4_distill_E/branch_E/distill_best_model.pth branch_E/seq4_best.pth
# Works locally but breaks on any machine with different directory structure
```

#### Correct Approach

```bash
# ✅ CORRECT: Copy the actual file, not a symlink
cp experiments/seq4_distill_E/branch_E/distill_best_model.pth \
  experiments/branch_E/seq4_distill_best_model.pth
git add experiments/branch_E/seq4_distill_best_model.pth
```

This adds ~8MB per checkpoint, which is acceptable for git. The alternative (symlinks) causes silent failures when the evaluation pipeline can't find the actual file.

**Rule of thumb**: If the file will be used on a different machine than where it was created, copy it. Symlinks are only safe within a single machine's filesystem.

### 6.4 NaN Detection

SSM layers are numerically sensitive. Monitor aggressively:

```bash
grep -c NaN/Inf training.log           # Count NaN events
grep "NaN\|Inf\|nan\|inf" training.log  # Find exact locations
```

If NaN detected after epoch 15-20: reduce learning rate.
If NaN detected immediately: check input data for invalid values.

### 6.2 Gradient Norm Tracking

Add gradient norm logging to detect instability early:
```python
total_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
```

Sustained gradient norm >10× baseline = impending divergence.

### 6.3 Action Output Diagnostics

For regression tasks (velocity prediction, etc.), track output distribution:
```python
action_mag = output.abs().mean().item()
action_var = output.var(dim=0).mean().item()
action_max_vz = output[:, 2].abs().max().item()  # vertical max
```

| Symptom | Meaning |
|---------|---------|
| `action_mag` < 0.1 | Mode collapse — model outputs near-zero |
| `action_var` < 0.001 | Output diversity too low |
| `action_max_vz` > 1.5 | Potential panic response near obstacles |

### 6.4 Data Loading Debugging

When training stalls with zero GPU activity:
```bash
# Check if process is alive
ps aux | grep train_script

# Check if data loading is the bottleneck
# Add timing to __getitem__:
start = time.time()
sample = self._load_sample(idx)
load_time = time.time() - start
if load_time > 0.5:
    print(f"Slow load: {idx} in {load_time:.2f}s")
```

---

## 7. Known Failure Modes & Fixes

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| NaN loss at epoch 15-20 | LR too high for SSM | Reduce `lr` to 1e-4, `clip_grad_norm` to 0.5 |
| Process dies silently | Parent shell kills child | Use `setsid` with redirected I/O |
| Dataloader hangs | Multiprocessing deadlock | Set `num_workers=2` or 0 |
| Disk full mid-training | Checkpoint + log files accumulate | `df -h /` and clean `opencode/log/` (can reach 16GB+) |
| torch.compile model can't load | `_orig_mod.` key prefix | Strip prefix: `{k.replace('_orig_mod.', ''): v}` |
| Simulator model loading fails | Architecture/checkpoint mismatch | Verify `run_competition.py` config matches training config |
| GPU utilization <5% | Tiny model, CPU data bottleneck | Use `num_workers=2`, enable TF32, try MPS for multi-experiment |
| SS2D CUDA OOM under concurrent load | SS2D (2D selective scan) requires large contiguous GPU memory. Under multi-process concurrency, PyTorch's allocator fragments VRAM. Even with 1.9GB free, no single 2GB contiguous block available. | **Not a memory leak — fragmentation issue.** Run SS2D-based models (VMamba, Branch A) alone, not alongside other training processes. Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or set `batch_size=32` to reduce allocation size. |
| Checkpoint dimension mismatch | Model code updated, weights old | Retrain or revert architecture change |

---

## 8. Checkpoint & Model Weight Management

### Saving
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'val_loss': val_loss,
    'config': training_config,
}, 'checkpoint.pth')
```

### Loading
```python
ckpt = torch.load('checkpoint.pth', map_location='cpu')
sd = ckpt.get('model_state_dict', ckpt)
# Strip compile prefix
if any(k.startswith('_orig_mod.') for k in sd.keys()):
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)  # Use strict=False for partial loading
```

---

## 9. Project-Specific References

### Vitfly Model Branches

| Branch | Architecture | Visual Encoder | Temporal Head | Params | Distill 60m Result |
|--------|-------------|----------------|---------------|--------|-------------------|
| A | VMamba + LSTM | SS2D (Mamba) | LSTM | 0.97M | 3 crashes |
| B | MambaVision + SSM | Hybrid (Attn+SSM) | Mamba-2 | 2.61M | 2 crashes (rescued) |
| B+ | MambaVision + Mamba-3 | Hybrid (Attn+SSM) | Mamba-3 | 2.55M | **1 crash** 🏆 |
| C | CNN + Mamba-3 | CNN | Mamba-3 | 2.41M | 3 crashes |
| D | STH-Mamba | CNN-like | Mamba-2 | 2.60M | 2 crashes |
| E | DecisionMamba | **Pure SSM** | **SSM** | 2.19M | **1 crash** 🏆 |

**Key finding (vitfly-specific)**: SSM must be present in BOTH visual encoder and temporal head for optimal distillation. Best results from pure-SSM (E) or hybrid-SSM (B+) architectures.

### Vitfly Quick Commands

```bash
# Full training (BC)
setsid python3 -u training/train_mamba_optimized.py \
  --data_dir training/datasets/data_full --branches A B C D E \
  --epochs 100 --batch_size 32 --lr 0.0001 --num_workers 2 \
  --clip_grad_norm 0.5 --seed 42 --val_split 0.2 \
  </dev/null >training/logs/full_run.log 2>&1 &

# Distillation training
setsid python3 -u training/train_distill.py \
  --branch E --epochs 50 --batch-size 32 --num-workers 2 \
  --alpha 1.0 --beta 1.0 --gamma 1.0 \
  </dev/null >training/logs/distill.log 2>&1 &

# Born-again distillation
setsid python3 -u training/train_distill.py \
  --branch E --teacher-branch Bplus \
  --teacher-ckpt experiments/.../branch_Bplus/distill_best_model.pth \
  --epochs 50 \
  </dev/null >training/logs/bornagain.log 2>&1 &

# Multi-step sequence training
setsid python3 -u training/train_mamba_optimized.py \
  --branches E --sequence_length 16 --num_workers 0 \
  --epochs 100 \
  </dev/null >training/logs/seq16.log 2>&1 &

# Training with MPS parallelization
nvidia-cuda-mps-control -d
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33 python3 ... &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33 python3 ... &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33 python3 ... &
wait
echo quit | nvidia-cuda-mps-control

# Monitor
tail -f training/logs/*.log
nvidia-smi
grep NaN training/logs/*.log

