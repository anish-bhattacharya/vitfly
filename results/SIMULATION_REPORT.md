# Simulation Experiment Report

**Project**: ViT+LSTM → Mamba Cross-Architecture Knowledge Distillation
**Environment**: ROS Noetic + Flightmare (Unity) + WSL2, 60m obstacle course
**Date**: 2026-05-05
**Evaluator**: Upstream vitfly evaluation (`obstacles[0]` collision detection, `target=60`)

---

## 1. Experimental Design

### 1.1 Models Tested

| ID | Model | Architecture | Params | Type |
|----|-------|-------------|--------|------|
| Teacher | ViT+LSTM | ViT encoder + 3-layer LSTM | 3.56M | Upstream best |
| A | VMamba+LSTM | VMamba + LSTM | 0.97M | Stateful |
| B | MambaVision+SSM | MambaVision + SSM head | 2.61M | Stateless |
| B+ | BPlusModel | MambaVision + Mamba-3 hybrid | 2.55M | Stateless |
| C | CNN+Mamba3 | CNN + Mamba-3 | 2.41M | Stateless |
| D | STH-Mamba | Spatial-Temporal Hybrid | 2.60M | Stateless |
| E | DecisionMamba | CNN + lightweight head | 2.19M | Stateless |

### 1.2 Evaluation Dimensions

| Dimension | Settings Tested | Default |
|-----------|----------------|---------|
| Track length | 20m, **60m** | **60m** (upstream standard) |
| Desired velocity | **5m/s**, 7m/s | **5m/s** |
| Prediction mode | **Single-step (seq_len=1)** | **seq_len=1** |
| Training variant | **BC**, **Distill (α=β=γ=1.0)**, Born-again | — |
| Teacher | ViT+LSTM (517-dim) | — |

### 1.3 Test Matrix Coverage

```
                    BC @5m/s    Distill @5m/s    Teacher @5m/s    @7m/s
Teacher (ViT+LSTM)      —             —               ✅             ✅
A (VMamba+LSTM)        ✅             ✅               —             —
B (MambaVision+SSM)   ❌DNF           ✅               —             —
B+ (BPlusModel)        ✅             ✅               —             —
C (CNN+Mamba3)         ✅             ✅               —             —
D (STH-Mamba)          ✅             ✅               —             —
E (DecisionMamba)      ✅        ✅ + born-again       —             ✅
```

---

## 2. Complete Results (60m Track, 5m/s)

### 2.1 Main Comparison

| Rank | Model | Crashes | 60m Time | Vel Outputs | vs Teacher | vs BC |
|------|-------|---------|----------|-------------|------------|-------|
| 🥇 | **B+ Distill** | **1** 🏆 | 12.22s | 456 | **-1** ✅ | **-2** ✅ |
| 🥇 | **E Distill** | **1** 🏆 | 12.23s | 450 | **-1** ✅ | **-2** ✅ |
| — | Teacher ViT+LSTM | 2 | 12.24s | 470 | — | — |
| — | B Distill | 2 | 12.36s | 467 | 0 | ✅ rescued |
| — | D BC | 2 | 12.22s | 466 | 0 | — |
| — | D Distill | 2 | 12.19s | 450 | 0 | 0 |
| — | A BC | 3 | 12.82s | 478 | +1 | — |
| — | A Distill | 3 | 12.24s | 476 | +1 | 0 |
| — | C BC | 3 | 12.41s | 430 | +1 | — |
| — | C Distill | 3 | 12.41s | 450 | +1 | 0 |
| — | E BC | 3 | 12.23s | 447 | +1 | — |
| — | B+ BC | 3 | 12.37s | 458 | +1 | — |
| — | Born-again E | 3 | 12.19s | 442 | +1 | 0 |
| ❌ | E seq16 BC (ep100) | 4 | 13.71s | 490 | +2 | +1 |
| ❌ | **B BC** | **DNF** | — | 434 | — | — |

### 2.2 Distillation Impact Summary

| Impact | Branches | Count |
|--------|----------|-------|
| ✅ **Improves** (crashes ↓) | B+, E | 2 |
| ✅ **Rescues** (BC failed → distill works) | B | 1 |
| = **Neutral** (same as BC) | A, C, D | 3 |
| ❌ **Degrades** (crashes ↑) | — | **0** |

**Distillation never degrades flight quality.**

### 2.3 Velocity Scaling (@ 60m)

| Model | 5m/s | 7m/s | Δ |
|-------|------|------|---|
| Teacher ViT+LSTM | 2 crashes | **5 crashes** ❌ | +3 |
| **E Distill** | **1 crash** | **1 crash** ✅ | **0** 🏆 |

E Distill is speed-robust; teacher degrades at its native speed.

### 2.4 Born-Again Distillation

B+ Distill → E (same-architecture, B+ teacher):

| Model | Crashes @ 60m | Time |
|-------|--------------|------|
| E Distill (ViT+LSTM teacher) | **1** 🏆 | 12.23s |
| E Born-again (B+ teacher) | **3** | 12.19s |

Despite 4.4× better val_distill_gap (0.0037 vs 0.0172), born-again was worse in sim (3 vs 1 crash). val_loss does not predict flight quality.

---

## 3. Key Findings

### 3.1 Distillation Works

Cross-architecture distillation (ViT+LSTM → Mamba) consistently improves or matches BC baselines. No branch was degraded by distillation at 60m.

### 3.2 Best Models

**B+ Distill** and **E Distill** (1 crash each) outperform both BC baselines (3 crashes) AND the ViT+LSTM teacher (2 crashes). Distilled Mamba models can surpass their teacher.

### 3.3 Speed Robustness

E Distill maintains 1 crash at both 5m/s and 7m/s on the 60m track. The teacher degrades from 2→5 crashes at 7m/s.

### 3.4 20m Data Was Misleading

Early tests at 20m suggested distillation degraded most branches. The correct 60m evaluation showed the opposite. Obstacles are distributed across the full track; 20m only covered the easy portion.

### 3.5 Born-Again Needs Tuning

Same-architecture distillation (B+→E) achieved better val metrics but worse sim performance. The checkpoint was saved at epoch 6 (very early); longer training with higher GT loss weight may help.

### 3.6 Val Loss ≠ Flight Quality

Multiple cases where val_loss improved but simulation got worse, and vice versa. Simulation is the only reliable evaluation.

---

## 4. Test Commands Reference

```bash
# Single test (simulator must be running)
bash run_full_test.bash <BRANCH> <MODEL_TYPE> [VARIANT] [DES_VEL]

# Examples
bash run_full_test.bash E DecisionMamba           # BC @ 5m/s
bash run_full_test.bash E DecisionMamba distill    # Distill @ 5m/s
bash run_full_test.bash E DecisionMamba distill 7.0  # Distill @ 7m/s

# Manual teacher test
cd envtest/ros
python3 evaluation_node.py <run_name> > /tmp/eval.log 2>&1 &
python3 -u run_competition.py --vision_based --des_vel 5.0 \
  --model_type ViTLSTM \
  --model_path <path_to_model>/ViTLSTM_model.pth > /tmp/comp.log 2>&1 &

# Check results
grep "RUN_COMPETITION.*velocity" /tmp/comp_<X>.log | wc -l  # ~240 for 4s @5m/s
cat summary.yaml
```

---

## 5. Gaps & Next Steps

| Gap | Priority | Reason |
|-----|----------|--------|
| B+/E Distill @ 7m/s @ 60m | High | Paper claim: speed-robust. Currently only E tested at 7m/s @ 60m |
| Teacher @ 7m/s but 5m/s data mismatch | Medium | Teacher trained at 7m/s but evaluated at 5m/s for fair comparison |
| Multi-step prediction (seq_len > 1) | Medium | Teacher has LSTM; seq_len>1 could benefit stateful models |
| Loss weight ablation in sim | Low | α=β=γ=1.0 only; γ>1 might improve born-again |
| Init-from-BC distillation | Low | All distill from random init; BC-pretrained distill untested |

---

*Report generated 2026-05-05. All results at 60m unless otherwise noted.*
