# Branch F Experiment Matrix

> **Goal**: Systematically validate that Branch F's Lightweight CNN + Mamba-3 architecture achieves competitive obstacle avoidance performance with **70% fewer parameters** than the Teacher model (~3.56M) through controlled, incremental experiments.

## Motivation

### Why a Systematic Experiment Matrix?

Distillation (F3) alone cannot validate architecture superiority. If distillation performs well, we cannot distinguish whether the credit belongs to:

1. **The architecture itself** (CNN + Mamba-3 is inherently capable)
2. **The teacher's guidance** (LSTM hidden states compensate for weak features)

By running pure BC first, we isolate the **architecture contribution**. Then we add variables one at a time (augmentation → distillation → multi-frame), measuring the marginal benefit of each. This yields a clean ablation chain where every experiment answers exactly one question.

### Research Questions

| # | Question | Answered By |
|---|----------|-------------|
| RQ1 | Can a 1.05M param CNN+Mamba-3 match a 3.56M ViT+LSTM in pure BC? | F1-BC vs Teacher |
| RQ2 | Does data augmentation improve robustness without architectural change? | F2-BC-Aug vs F1-BC |
| RQ3 | Does teacher knowledge distillation add value beyond pure BC? | F3-Distill vs F2-BC-Aug |
| RQ4 | Does multi-frame temporal input improve trajectory quality? | F4-MultiSeq vs F1-BC |
| RQ5 | How does Branch F compare to similar-sized Branch C (~3.0M)? | F1-BC vs Branch C |

## Architecture

### Branch F Specs

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| Depth Encoder | Lightweight CNN (stem + 2 stages + head) | 496,608 |
| Metadata Concat | vel×0.1 + quaternion (7-dim) → 263-dim total | — |
| Temporal Head | Mamba-3 SSM × 2 blocks (d_state=64, hidden=256) | ~597,800 |
| Output Layer | SpectralNorm Linear(256 → 3) | 771 |
| **Total** | | **~1,095,179** |

### Comparison with Baselines

| Model | Encoder | Head | Total Params | MACs |
|-------|---------|------|-------------|------|
| Teacher (ViT+LSTM) | ViT-B/16 (~86M) | LSTM(256, 2-layer) | ~3.56M | ~17.6G |
| B+ (MambaVision+Mamba-3) | MambaVision (~11.2M) | Mamba-3(256, 2-layer) | ~11.8M | ~2.1G |
| Branch C (CNN+Mamba-3) | MobileNetV3-CNN (~1.8M) | Mamba-3(512) custom | ~3.0M | ~0.5G |
| **Branch F (Ours)** | **Lightweight CNN (~0.5M)** | **Mamba-3(256, 2-layer)** | **~1.1M** | **~0.2G** |

### Parameter Efficiency Ratios

```
Teacher (3.56M)  ─────────────────── 3.4× larger than F
Branch C (3.0M)  ────────────────── 2.9× larger than F
B+ (11.8M)       ────────────────── 11.2× larger than F
Branch F (1.1M)  ─── [SWEET SPOT for embedded deployment]
```

---

## Experiment Design

### Guiding Principle: Add One Variable at a Time

```
   Baseline     Augmentation     Distillation     Multi-Frame
  ┌─────────┐   ┌──────────┐    ┌───────────┐    ┌──────────┐
  │ F1-BC   │──▶│ F2-BC-Aug│───▶│ F3-Distill│ ──▶│ F4-Multi │
  │ (P0)    │   │ (P1)     │    │ (P2)      │    │ Seq (P1) │
  └─────────┘   └──────────┘    └───────────┘    └──────────┘
       │              │               │                │
  Pure BC          BC + aug      Teacher distill    Seq len 4-8
  No aug           No distill   BC + aug + distill  No aug/distill
```

---

### F1-BC: Pure Behavioral Cloning Baseline

**Priority**: P0 (MUST implement first)

**Status**: 🔴 Not started — implementation planned below

#### Hypothesis

> A Lightweight CNN + Mamba-3 architecture with only 1.05M parameters can match the Teacher ViT+LSTM (3.56M params) on behavioral cloning from expert demonstrations, achieving ≤5 crashes on Spheres 7m/s and ≤1 crash on Trees 7m/s. The Mamba-3 SSM temporal head compensates for the reduced encoder capacity through superior temporal modeling.

#### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 100 | Matches established training protocol |
| **Batch size** | 32 | Fits GPU memory; same as other branches |
| **Optimizer** | AdamW | Default for transformer-based training |
| **Learning rate** | 1e-3 | Cosine annealing from this peak |
| **Weight decay** | 1e-4 | Prevents overfitting on 1234 trajectories |
| **Warmup epochs** | 5 | Gradual LR ramp-up prevents early divergence |
| **Grad clipping** | 1.0 | Universal for transformers/SSMs |
| **LR schedule** | Cosine annealing | Decays to 0 over 100 epochs |
| **Loss function** | MSE on velocity (vx, vy, vz) | Standard BC objective |
| **Precision** | bf16 mixed precision | Faster training, stable gradients |
| **Data** | data_full (1234 trajectories) | Full dataset |
| **Augmentation** | None | Intentional — isolates architecture performance |
| **Distillation** | None | Intentional — pure BC only |
| **Val split** | 10% | 90/10 train/val (matches distill config) |

#### Success Criteria

| Scenario | Metric | Target | Compared To |
|----------|--------|--------|-------------|
| Spheres 5m/s | Crash count (10 runs) | ≤ 3 | Teacher: ≤3 |
| Spheres 7m/s | Crash count (10 runs) | ≤ 5 | Teacher: ≤5 |
| Trees 5m/s | Crash count (10 runs) | ≤ 0 | Teacher: ≤0 |
| Trees 7m/s | Crash count (10 runs) | ≤ 1 | Teacher: ≤1 |
| All | Trajectory smoothness | Comparable to Teacher | Qualitative |
| All | Control effort | Comparable to Teacher | Qualitative |
| **Primary** | **Parameter efficiency** | **70% fewer than Teacher** | **~1.1M vs ~3.56M** |

#### Ablation Contribution

F1-BC establishes the **architecture baseline**. All subsequent experiments measure marginal improvement over this point.

---

### F2-BC-Aug: Behavioral Cloning with Data Augmentation

**Priority**: P1 (implement after F1-BC converges)

**Status**: 🟡 Not started

#### Hypothesis

> Adding data augmentation (horizontal flip + depth noise) to F1-BC improves generalization and reduces overfitting, leading to fewer crashes in simulation. All other factors held constant.

#### Training Configuration

| Parameter | F1-BC Value | F2-BC-Aug Value | Delta |
|-----------|-------------|-----------------|-------|
| Augmentation | None | Horizontal flip (p=0.5) + depth noise (σ=0.02, p=0.3) | **Changed** |
| All others | Same as F1-BC | Same as F1-BC | Unchanged |

#### Success Criteria

| Scenario | Metric | Target | Compared To |
|----------|--------|--------|-------------|
| Spheres 7m/s | Crash count (10 runs) | **Fewer** than F1-BC | F1-BC result |
| Trees 7m/s | Crash count (10 runs) | ≤ 1 | F1-BC result |
| All | Generalization to unseen scenarios | **Better** than F1-BC | Qualitative |

#### Ablation Contribution

Measures the **isolated effect of data augmentation** on a fixed architecture. If F2-BC-Aug significantly outperforms F1-BC, augmentation is a critical component. If not, the architecture is already data-efficient.

---

### F3-Distill: Progressive Hierarchical Distillation

**Priority**: P2 (implement after F1-BC and F2-BC-Aug)

**Status**: ✅ Already implemented — `train_branch_f_distill.py` + `configs/branch_f_distill_config.yaml`

#### Hypothesis

> Progressive Hierarchical Distillation (3-stage: encoder → temporal head → end-to-end) transfers knowledge from the Teacher ViT+LSTM to Branch F, further improving performance beyond pure BC. The gap between F3-Distill and F2-BC-Aug measures the value of teacher guidance.

#### Training Configuration

Already documented in `configs/branch_f_distill_config.yaml`. 3 stages:

| Stage | Name | Epochs | Trainable | LR |
|-------|------|--------|-----------|-----|
| 1 | Encoder Distillation | 50 | CNN encoder only (feature alignment) | 1e-3 |
| 2 | Temporal Head Distillation | 30 | Mamba-3 head only (hidden + action) | 5e-4 |
| 3 | End-to-End Fine-Tuning | 20 | All student params (joint optimization) | 1e-4 |

#### Success Criteria

| Scenario | Metric | Target | Compared To |
|----------|--------|--------|-------------|
| Spheres 7m/s | Crash count (10 runs) | **Fewer** than F2-BC-Aug | F2-BC-Aug result |
| Trees 7m/s | Crash count (10 runs) | ≤ 1 | F2-BC-Aug result |
| All | Should approach B+ performance | Within 10% of B+ crashes | B+ baseline |

#### Ablation Contribution

Measures the **value added by teacher knowledge transfer** beyond pure BC + augmentation. This is the key experiment that justifies the distillation pipeline.

---

### F4-MultiSeq: Multi-Sequence Temporal Modeling

**Priority**: P1 (can run in parallel with F2 after F1-BC)

**Status**: 🟡 Not started

#### Hypothesis

> Mamba-3's SSM temporal modeling benefits from longer input sequences (4-8 frames) compared to single-frame inference, producing smoother trajectories with fewer oscillations. Unlike F1-F3 (single-frame), this variant processes temporal windows.

#### Training Configuration

| Parameter | F1-BC Value | F4-MultiSeq Value | Delta |
|-----------|-------------|-------------------|-------|
| Sequence length | 1 | 4 | **Changed** |
| Batch size | 32 | 8 (adjust for GPU memory) | **Changed** |
| Augmentation | None | None | Unchanged |
| Distillation | None | None | Unchanged |
| Data loader | `create_lazy_dataloader` | `create_sequence_dataloader` | **Changed** |
| All other params | Same as F1-BC | Same as F1-BC | Unchanged |

#### Experimental Variants

> Note: F4 is most meaningful AFTER F1-BC, since the Mamba-3 head processes single frames in F1-F3. The multi-sequence variant tests whether the SSM's temporal recurrence provides additional benefit.

| Sub-variant | Seq Len | Batch | Rationale |
|-------------|---------|-------|-----------|
| F4a | 4 | 16 | Minimal temporal context |
| F4b | 8 | 8 | Full temporal window |
| F4c (if needed) | 16 | 4 | Extended temporal window |

#### Success Criteria

| Metric | Target | Compared To |
|--------|--------|-------------|
| Trajectory smoothness | **Better** (lower jerk) than F1-BC | F1-BC |
| Control effort variance | **Lower** than F1-BC | F1-BC |
| Crash count | **No degradation** from F1-BC | F1-BC |
| Oscillation frequency (qualitative) | **Fewer** corrective zigzags | F1-BC |

#### Ablation Contribution

Tests whether Mamba-3's **temporal SSM capabilities are underutilized** with single-frame input. Positive result validates the Mamba-3 design choice over simpler temporal heads.

---

## Experiment Dependency Graph

```
F1-BC (P0)
  ├── establishes baseline ──→ ALL comparisons reference this
  │
  ├── F2-BC-Aug (P1) ── add augmentation → measures aug benefit
  │     │
  │     └── F3-Distill (P2) ── add distillation on top of aug
  │
  └── F4-MultiSeq (P1) ── parallel track, same baseline
        (can run concurrently with F2)
```

**Sequential constraints**:
- F2-BC-Aug depends on F1-BC completion (shares config + protocol)
- F3-Distill depends on F2-BC-Aug completion (measures marginal gain)
- F4-MultiSeq is independent of F2/F3 (different data pipeline)

---

## Comparison Baselines

All baselines have **existing results** — no retraining needed.

### Teacher (ViT + LSTM) — ~3.56M params

| Scenario | Expected Crashes (10 runs) | Source |
|----------|---------------------------|--------|
| Spheres 5m/s | ≤ 3 | Paper results |
| Spheres 7m/s | ≤ 5 | Paper results |
| Trees 5m/s | ≤ 0 | Paper results |
| Trees 7m/s | ≤ 1 | Paper results |

### B+ (MambaVision + Mamba-3) — ~11.8M params

| Scenario | Expected Crashes (10 runs) | Source |
|----------|---------------------------|--------|
| Spheres 5m/s | TBD | Branch B+ eval |
| Spheres 7m/s | TBD | Branch B+ eval |
| Trees 5m/s | TBD | Branch B+ eval |
| Trees 7m/s | TBD | Branch B+ eval |

### Branch C (CNN + Mamba-3) — ~3.0M params

| Scenario | Expected Crashes (10 runs) | Source |
|----------|---------------------------|--------|
| Spheres 5m/s | TBD | Branch C eval |
| Spheres 7m/s | TBD | Branch C eval |
| Trees 5m/s | TBD | Branch C eval |
| Trees 7m/s | TBD | Branch C eval |

---

## Evaluation Protocol

### Simulation Scenarios

| Scenario | World | Speed | Difficulty | Duration |
|----------|-------|-------|------------|----------|
| S1 | Spheres | 5 m/s | Easy | ~60s |
| S2 | Spheres | 7 m/s | Medium | ~60s |
| S3 | Trees | 5 m/s | Medium | ~60s |
| S4 | Trees | 7 m/s | Hard | ~60s |

### Metrics

| Metric | Definition | Primary For |
|--------|------------|-------------|
| **Crash count** | Number of collisions per 10 runs | All experiments |
| **Crash rate** | % of runs with ≥1 collision | All experiments |
| **Trajectory smoothness** | Mean squared jerk (3rd derivative of position) | F4-MultiSeq |
| **Control effort** | Sum of squared velocity commands | F4-MultiSeq |
| **Completion rate** | % of runs finishing without crash | All experiments |
| **Mean trajectory duration** | Average time to complete course | All experiments |

### Statistical Rigor

- **10 independent runs** per (scenario, model) pair
- **Different random seeds** for each run
- Report: **mean ± std** for all metrics
- Statistical significance: **Welch's t-test** (p < 0.05) between experiment pairs
- Effect size: **Cohen's d** for pairwise comparisons

### Primary Comparison Matrix

```
             ┌──────┬──────┬──────┬──────┬──────┬──────┐
             │Teacher│ B+   │ Br.C │ F1   │ F2   │ F3   │
             │3.56M  │11.8M │ 3.0M │ 1.1M │ 1.1M │ 1.1M │
┌────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│Spheres 5m/s│  X   │  X   │  X   │  X   │  X   │  X   │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│Spheres 7m/s│  X   │  X   │  X   │  X   │  X   │  X   │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│Trees 5m/s  │  X   │  X   │  X   │  X   │  X   │  X   │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│Trees 7m/s  │  X   │  X   │  X   │  X   │  X   │  X   │
└────────────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

### Ablation Isolation

| Comparison | Variable Isolated | Confounders |
|------------|------------------|-------------|
| F1-BC vs Teacher | Architecture (CNN+Mamba-3 vs ViT+LSTM) | Parameter count (1.1M vs 3.56M) |
| F2-BC-Aug vs F1-BC | Data augmentation | None — single variable change |
| F3-Distill vs F2-BC-Aug | Teacher distillation | None — same aug, added distill |
| F4-MultiSeq vs F1-BC | Sequence length | Data loader difference |
| F1-BC vs Branch C | Encoder design (Lightweight vs MobileNetV3) | Parameter count (1.1M vs 3.0M) |

---

## Implementation Plan: F1-BC

### Overview

F1-BC requires two changes:
1. **Add Branch F to `train_mamba_optimized.py`** — import the model, register the branch
2. **Create training config** — `branch_f_bc_baseline.yaml`

The training script `train_mamba_optimized.py` currently supports branches A, B, C, D, E, Bplus, Fusion, Essm but **not Branch F**. We need to add it.

### Step 1: Add Branch F Model Import

**File**: `/root/vitfly/training/train_mamba_optimized.py`

Add the import path and model factory:

```python
# After existing imports (~line 43-44)
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_F_lightweight_mamba3/models')

# After existing model imports (~line 55)
from branch_f_model import BranchFModel, create_branch_f_model
```

### Step 2: Register Branch F in `create_model()`

In the `create_model()` function (~line 159), add:

```python
elif branch_name == 'F':
    model = create_branch_f_model(config)
```

### Step 3: Add Branch F to valid branch check

In `main()` (~line 535), update the valid branches check:

```python
if branch not in ['A', 'B', 'C', 'D', 'E', 'Bplus', 'Fusion', 'Essm', 'F']:
```

### Step 4: Add Branch F Config

The `train_branch()` function uses a hardcoded config dict (lines 308-322). This config won't work for Branch F since it uses different architecture parameters. We need to either:

**Option A (Recommended): Modify `train_branch()` to accept branch-specific configs**

Add a branch-specific config mapping before the config dict:

```python
# Branch-specific model configs
branch_configs = {
    'A': { ... existing defaults ... },
    'B': { ... },
    'C': { ... },
    'D': { ... },
    'E': { ... },
    'Bplus': { ... },
    'F': {
        'cnn_encoder_config': {
            'in_channels': 1,
            'output_dim': 256,
        },
        'mamba3_d_state': 64,
        'mamba3_hidden': 256,
        'mamba3_layers': 2,
        'mamba3_headdim': 32,
        'mamba3_chunk_size': 32,
        'dropout': 0.1,
    }
}

config = branch_configs.get(branch_name, branch_configs['B'])  # Default to B-style config
```

### Step 5: Run Training

```bash
cd /root/vitfly/training
python train_mamba_optimized.py \
    --branches F \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --warmup_epochs 5 \
    --data_dir /root/vitfly/training/datasets/data_full \
    --save_dir /root/vitfly/experiments/mamba_branches/branch_F_lightweight_mamba3/checkpoints \
    --seed 42
```

### Step 6: Validate

Monitor training curves (loss should converge smoothly, val loss < 0.001). Check for:
- No NaN losses
- Gradient norms stable (< 10.0)
- Validation loss not diverging from training loss
- Convergence within 100 epochs

### Training Config File

The config file at `/root/vitfly/training/configs/branch_f_bc_baseline.yaml` serves as the single source of truth for F1-BC hyperparameters (see adjacent file).

---

## Appendix: Success Criteria Summary Table

| Experiment | Primary Metric | Target | Compared To | Priority |
|------------|---------------|--------|-------------|----------|
| F1-BC | Spheres 7m/s crashes | ≤ 5 | Teacher (≤5) | **P0** |
| F1-BC | Trees 7m/s crashes | ≤ 1 | Teacher (≤1) | **P0** |
| F1-BC | Parameter count | ~1.1M (70% fewer) | Teacher 3.56M | **P0** |
| F2-BC-Aug | Spheres 7m/s crashes | < F1-BC | F1-BC baseline | P1 |
| F3-Distill | Spheres 7m/s crashes | < F2-BC-Aug | F2-BC-Aug | P2 |
| F4-MultiSeq | Trajectory smoothness | Better jerk metric | F1-BC | P1 |

## Appendix: Quick Reference — Key Files

| File | Purpose |
|------|---------|
| `branch_F_lightweight_mamba3/models/__init__.py` | Model exports |
| `branch_F_lightweight_mamba3/models/branch_f_model.py` | Full Branch F model |
| `branch_F_lightweight_mamba3/models/lightweight_cnn_encoder.py` | CNN encoder |
| `branch_F_lightweight_mamba3/models/mamba3_head.py` | Mamba-3 SSM head |
| `branch_F_lightweight_mamba3/EXPERIMENTS.md` | **This file** — experiment design |
| `training/configs/branch_f_bc_baseline.yaml` | F1-BC training config |
| `training/configs/branch_f_distill_config.yaml` | F3-Distill training config |
| `training/train_mamba_optimized.py` | Training script (needs F added) |
| `training/train_branch_f_distill.py` | Distillation training script |
| `training/lazy_dataloading.py` | Data loading (single-frame + sequence) |
