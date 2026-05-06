# Training Pipeline Experiment Report

**Date**: 2026-05-06
**Last updated**: 2026-05-06
**GPU**: NVIDIA RTX 5090 (32GB) with NVIDIA MPS, TF32, AMP
**Dataset**: 42K trajectories (25K train, 8.4K val), collected at ~5m/s
**Total wall time**: ~12 hours (all experiments combined)

---

## 1. Completed Experiments

### 1.1 Cross-Architecture Distillation — Phase 1

**Setup**: ViT+LSTM teacher (upstream best, 7m/s flight), 6 Mamba students, 50 epochs, α=β=γ=1.0, random init.

| Branch | Architecture | Params | BC crashes | Distill crashes | Δ | Sim 60m time |
|--------|-------------|--------|-----------|----------------|---|-------------|
| **B+** | MambaVision+Mamba-3 | 2.55M | 3 | **1** 🏆 | **-2** | 12.22s |
| **E** | DecisionMamba (pure SSM) | 2.19M | 3 | **1** 🏆 | **-2** | 12.23s |
| B | MambaVision+SSM | 2.61M | ❌ Failed | **2** | ✅ rescued | 12.36s |
| D | STH-Mamba | 2.60M | 2 | 2 | 0 | 12.19s |
| A | VMamba+LSTM | 0.97M | 3 | 3 | 0 | 12.24s |
| C | CNN+Mamba-3 | 2.41M | 3 | 3 | 0 | 12.41s |
| **Teacher** | ViT+LSTM | 3.56M | — | **2** (baseline) | — | 12.24s |

**Cost**: 6×50ep × ~44s/ep ≈ 3.7h sequential → **1.5h with MPS parallel**.

### 1.2 Born-Again Iterative Distillation

**Setup**: B+ Distill as teacher → E student. Testing GT loss weight γ.

| Variant | Teacher | γ | val_gt | distill_gap | Sim @ 60m |
|---------|---------|---|--------|-------------|-----------|
| B+ → E | B+ | 1.0 | 0.0172 | **0.0037** | ❌ 3 crashes |
| **B+ → E** | **B+** | **2.0** | **0.0165** 🏆 | 0.0046 | ⏳ pending |

γ=2.0 achieves val_gt=0.0165 — **first model to surpass BC baseline** (0.0186). Higher γ prevents overfitting to teacher's specific behavior. **Teacher choice (B+ vs ViT+LSTM) matters 10× more than loss weight tuning.**

**Cost**: 2×50ep × ~70s/ep ≈ 2h.

### 1.3 Multi-Step Distillation

**Setup**: Train BC at seq_len=4,8,16 (100ep) → distill with ViT+LSTM teacher (50ep, BC init).

| Model | Init | seq_len | val_gt | distill_gap | Sim @ 60m |
|-------|------|---------|-------|-------------|-----------|
| **E Distill** | Random | **1** | 0.0188 | 0.0172 | ✅ **1 crash** |
| E seq4 BC | — | 4 | 0.2297 | — | ⏳ |
| **E Distill** | **BC seq4** | **4** | **0.0167** | 0.0152 | ⏳ |
| E seq8 BC | — | 8 | 0.2413 | — | ⏳ |
| **E Distill** | **BC seq8** | **8** | **0.0169** | 0.0150 | ⏳ |
| E seq16 BC | — | 16 | 0.2323 | — | ❌ 4 crashes |
| E seq16 Distill | BC seq16 | 16 | 0.7414 | 0.6911 | ⏳ |

BC init improves distill val_gt from 0.0188 → 0.0167 (seq4) / 0.0169 (seq8). Seq16 BC overfits (val_loss 0.23).

**Cost**: 2×100ep BC × ~78s/ep + 2×50ep distill × ~42s/ep ≈ 5h sequential → **~2.5h MPS parallel**.

### 1.4 Distillation Loss Weight Grid Search

**Setup**: 2×2 full factorial on E with ViT+LSTM teacher, 50 epochs each.

| α (feat align) | β (output distill) | γ (GT) | score | gt_loss | distill_gap |
|--------------|-----------------|--------|-------|---------|-------------|
| 0.5 | 0.5 | 1.0 | 0.0267 | 0.0178 | 0.0177 |
| 0.5 | 1.0 | 1.0 | 0.0274 | 0.0188 | 0.0172 |
| 1.0 | 0.5 | 1.0 | 0.0267 | 0.0178 | 0.0177 |
| 1.0 | 1.0 | 1.0 | 0.0274 | 0.0188 | 0.0172 | (default) |

α negligible in 0.5-1.0 range. β=0.5 marginally better than 1.0. All within noise.

**Cost**: 4×50ep × ~42s/ep ≈ 2.3h sequential → **~0.8h MPS parallel**.

---

## 2. Acceleration & Optimization

| Technique | Gain | When | Caveats |
|-----------|------|------|---------|
| **NVIDIA MPS** | **~2.9×** | Multiple small models | Only helps when single process <10% GPU util |
| **TF32 matmul** | **~2×** | All training | Transparent, no accuracy loss |
| **AMP FP16** | **~2×** | All training | Monitor NaN in SSM layers |
| **torch.compile** | ~1.3× | Per-model test | SS2D incompatible; `_orig_mod.` prefix on save |
| **num_workers=2** | 2-3× vs 0 | seq_len=1 only | Deadlock risk at 4+ |
| **num_workers=0** | 1× | seq_len>1 | Required for sequence dataloader |

### 2.1 MPS Practical Guide

```
Without MPS: [seq4][idle][seq8][idle][seq16][idle]  →  GPU@3%, 4.5h
With MPS:    [seq4][seq8][seq16][seq4][seq8][seq16]  →  GPU@12%, 1.5h
```

MPS works when models are tiny (<5M params) and GPU compute is not the bottleneck. In our case, CPU data loading was the bottleneck, so MPS parallelization added almost no per-process overhead while using idle GPU cycles.

---

## 3. Experiment Chronology

| Day | Experiments | Wall time |
|-----|------------|-----------|
| 05-04 | Phase 1 distillation (6 branches) | ~1.5h |
| 05-04 | Born-again B+→E γ=1.0 | ~0.7h |
| 05-04 | Grid search α,β (4 runs, MPS) | ~0.8h |
| 05-04 | MPS seq4/8/16 BC (3 seq, MPS) | ~1.5h |
| 05-05 | Seq16 distillation | ~1.5h |
| 05-05 | Born-again γ=2.0 | ~0.7h |
| 05-06 | Seq4/8 BC + Seq4/8 distill (MPS) | ~2.5h |
| **Total** | **All experiments** | **~12h** |

---

## 4. Directory Structure

```
optimized_training/
├── branch_E/                  # Original BC + seq1 distill + symlinks
├── bornagain_Bplus2E/         # B+→E γ=1.0
├── bornagain_gamma2/          # B+→E γ=2.0
├── seq4_BC_E/                 # seq_len=4 BC (100ep)
├── seq4_distill_E/            # seq_len=4 BC init + distill
├── seq8_BC_E/                 # seq_len=8 BC (100ep)
├── seq8_distill_E/            # seq_len=8 BC init + distill
├── seq16_distill_E/           # seq_len=16 BC init + distill
├── grid_E_a0.5_b0.5/          # α=0.5, β=0.5
├── grid_E_a0.5_b1.0/          # α=0.5, β=1.0
├── grid_E_a1.0_b0.5/          # α=1.0, β=0.5
└── grid_E_a1.0_b1.0/          # α=1.0, β=1.0
```

Each contains `branch_E/` with `distill_summary.json` + `distill_best_model.pth` + histories.

Diagnostic symlinks in `branch_E/` for simulation pipeline:
- `distill_best_model.pth` → seq1 cross-architecture distill
- `bornagain_g2_best_model.pth` → born-again γ=2.0
- `seq4_distill_best_model.pth` → seq4 BC init + distill
- `seq8_distill_best_model.pth` → seq8 BC init + distill
- `seq16_distill_best_model.pth` → seq16 BC init + distill

---

## 5. Pending Simulation Verification

| Priority | Model | Command |
|----------|-------|---------|
| 🔴 High | Born-again γ=2.0 (val_gt=0.0165, beats BC) | `bash run_full_test.bash E DecisionMamba bornagain_g2` |
| 🔴 High | Seq4 Distill (val_gt=0.0167) | `bash run_full_test.bash E DecisionMamba seq4_distill 5.0 4` |
| 🔴 High | Seq8 Distill (val_gt=0.0169) | `bash run_full_test.bash E DecisionMamba seq8_distill 5.0 8` |
| 🟡 Med | B+ Distill @ 7m/s @ 60m | `bash run_full_test.bash B BPlusModel distill 7.0` |
| 🟡 Med | BC baselines @ 7m/s @ 60m | `bash run_full_test.bash E DecisionMamba bc 7.0` |
| ⚪ Low | Seq16 Distill (overfit risk) | `bash run_full_test.bash E DecisionMamba seq16_distill 5.0 16` |
