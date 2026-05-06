# Training Pipeline Experiment Report

**Date**: 2026-05-06
**GPU**: NVIDIA RTX 5090 (32GB) with NVIDIA MPS
**Dataset**: 42K trajectories (25K train, 8.4K val)

---

## 1. Completed Experiments

### 1.1 Cross-Architecture Distillation (Phase 1)

All 6 Mamba branches distilled from ViT+LSTM teacher (50 epochs, α=β=γ=1.0):
- B+ (BPlusModel): **1 crash @ 60m** 🏆
- E (DecisionMamba): **1 crash @ 60m** 🏆
- B (MambaVision+SSM): 2 crashes (rescued from BC failure)
- A, C, D: Neutral (match BC)

### 1.2 Born-Again Iterative Distillation

| Variant | Teacher | γ | val_gt | distill_gap | Sim @ 60m |
|---------|---------|---|--------|-------------|-----------|
| B+ → E | B+ | 1.0 | 0.0172 | **0.0037** | 3 crashes |
| **B+ → E** | **B+** | **2.0** | **0.0165** 🏆 | 0.0046 | ⏳ pending |

γ=2.0 achieves the first model to surpass BC baseline (0.0186) on val_gt.

### 1.3 Multi-Step Distillation

| Model | BC Init | seq_len | val_gt | distill_gap | Sim |
|-------|---------|---------|--------|-------------|-----|
| **E Distill** | Random | **1** | 0.0188 | 0.0172 | **1 crash** |
| **E Distill** | **BC seq4** | **4** | **0.0167** | 0.0152 | ⏳ pending |
| **E Distill** | **BC seq8** | **8** | **0.0169** | 0.0150 | ⏳ pending |
| E seq16 BC | — | 16 | 0.2323 | — | 4 crashes |
| E seq16 Distill | BC seq16 | 16 | 0.7414 | 0.6911 | ⏳ pending |

BC pretraining improves distill val_gt. Seq16 BC overfits.

### 1.4 Distillation Loss Weight Grid Search

α (feature alignment) and β (output distill) on E with ViT+LSTM teacher:

| α | β | score | gt | distill |
|---|---|-------|----|---------|
| 0.5 | 0.5 | 0.0267 | 0.0178 | 0.0177 |
| 1.0 | 1.0 | 0.0274 | 0.0188 | 0.0172 |

α has negligible effect in 0.5-1.0. β=0.5 marginally better.

---

## 2. Acceleration & Optimization

| Technique | Gain | Notes |
|-----------|------|-------|
| **NVIDIA MPS** | ~2.9× | 4 parallel runs: 4.4h → 1.5h |
| **TF32** | ~2× | Transparent, no accuracy loss |
| **AMP (FP16)** | ~2× | Monitor for NaN in SSM layers |
| **num_workers=2** | 2-3× vs 0 | seq_len=1 only; seq>1 requires 0 |

---

## 3. Directory Structure

```
optimized_training/
├── branch_E/                  # Original BC + seq1 distill
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

---

## 4. Pending Simulation Verification

- [ ] Born-again γ=2.0 @ 60m
- [ ] Seq4 Distill @ 60m
- [ ] Seq8 Distill @ 60m
- [ ] Seq16 Distill @ 60m
- [ ] B+ Distill @ 7m/s @ 60m
- [ ] BC baselines at 7m/s @ 60m
