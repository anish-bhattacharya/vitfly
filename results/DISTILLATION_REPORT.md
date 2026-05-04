# Distillation Experiment Report

**Cross-Architecture Knowledge Distillation: ViT+LSTM → Mamba**
Date: 2026-05-04
Based on literature survey in `literature/survey.md`

---

## 1. Motivation

Behavior cloning (BC) from scratch on each Mamba branch independently ignores the
knowledge already captured by the upstream ViT+LSTM teacher (best model, 7m/s real flight).
Knowledge distillation offers a way to transfer this pre-trained expertise into Mamba
architectures, potentially outperforming from-scratch BC.

### Why Cross-Architecture Distillation Is Hard

Naive output-level distillation (matching student logits to teacher logits) **fails** for
cross-architecture transfer. MOHAWK (NeurIPS 2024) explicitly proved this: when teacher and
student have fundamentally different sequence mixing mechanisms (attention vs SSM),
only multi-stage alignment at matrix → hidden-state → output levels succeeds.

---

## 2. Literature-Supported Design

### 2.1 Teacher Model
- **ViT+LSTM** (upstream best, `models/ViTLSTM_model.pth`, ~14MB)
- Loaded in eval mode, frozen during distillation
- Input: depth image → ViT encoder → LSTM → velocity command

### 2.2 Student Architectures (6 variants)

| Branch | Model | Params | BC Val Loss | BC Epochs |
|--------|-------|--------|-------------|-----------|
| A | VMamba + LSTM | 974K | 0.0161 | 27 |
| B | MambaVision + SSM | 2.61M | 0.0205 | 4 |
| Bplus | MambaVision + Mamba-3 | 2.55M | 0.0231 | 1 |
| C | CNN + Mamba-3 | 2.41M | 0.0221 | 1 |
| D | STH-Mamba | 2.60M | 0.0173 | 1 |
| E | DecisionMamba | 2.19M | 0.0186 | 1 |

**Data source**: Actual BC checkpoints from `experiments/mamba_branches/optimized_training/`.

### 2.3 Distillation Strategy (Three-Phase, MOHAWK-inspired)

```
Phase 1: Encoder Feature Alignment
    ┌──────────────┐    ┌──────────────────┐
    │ ViT Encoder   │    │ Mamba Encoder    │
    │ (teacher, fz) │    │ (student, train) │
    └──────┬───────┘    └───────┬──────────┘
           │                    │
           └────────┬───────────┘
                    ▼
         L_feat = MSE(f_teacher, f_student)
         
    MOHAWK Stage 1-2 analog: align intermediate representations
    X-Distill analog: feature-level supervision on encoder output

Phase 2: Output Distillation (Logit Level)
    ┌──────────────┐    ┌──────────────────┐
    │ ViT+LSTM      │    │ Mamba + Head     │
    │ (teacher, fz) │    │ (student, train) │
    └──────┬───────┘    └───────┬──────────┘
           │                    │
           ├── v_teacher ───────┤
           │                    │
           └────────┬───────────┘
                    ▼
         L_distill = MSE(v_teacher, v_student)
    
    MOHAWK Stage 3 analog: end-to-end distillation
    Temperature-scaled soft targets for smoother gradients

Phase 3: Joint Training (Distill + GT)
    L_total = α · L_feat + β · L_distill + γ · L_gt

    where L_gt = MSE(v_student, v_ground_truth)
```

### 2.4 Hyperparameter Search Space

| Param | Values | Notes |
|-------|--------|-------|
| α (feat weight) | 0.0, 0.1, 0.5, 1.0 | 0.0 = no feature alignment (ablation) |
| β (distill weight) | 0.0, 0.5, 1.0, 2.0 | 0.0 = BC only (ablation) |
| γ (GT weight) | 0.5, 1.0 | Always present for grounding |
| T (temperature) | 1.0, 2.0, 4.0 | Soften teacher logits |
| Encoder alignment layer | last, all, adaptive | Which layer to align |

### 2.5 Ablation Experiments

**Design**: All ablation experiments use the SAME branch (default B) to isolate loss component effects.
Cross-architecture comparison is a separate axis (see §3 Q2).

**Implementation**: Two tracks exist via `--init-from-bc` flag:
- **Track 1 (from scratch)**: Student random init, cleanest causal attribution
- **Track 2 (from BC)**: Load branch_{X}/best_model.pth, test practical improvement

| ID | α | β | γ | What It Tests |
|----|---|---|---|---------------|
| C0 | 0 | 0 | 1 | Pure BC baseline (from scratch) |
| C1 | 1 | 0 | 1 | Feature alignment only |
| C2 | 0 | 1 | 1 | Output distillation only |
| C3 | 1 | 1 | 1 | Full multi-stage (MOHAWK-style) |
| C4 | 0.5 | 1 | 1 | Tuned feature weight |
| C5 | 1 | 0.5 | 1 | Tuned distill weight |

---

## 3. Scientific Questions

### Q1: Can cross-architecture knowledge (ViT→Mamba) transfer?
X-Distill shows ViT→CNN works for robotics. MOHAWK shows Transformer→Mamba works for NLP.
Our experiment bridges these: **first test of ViT+LSTM→Mamba for robot control.**

**Prediction**: Yes, with proper multi-stage alignment. Feature alignment alone (C1) will
improve over BC baseline. Full multi-stage (C3) will give best results.

### Q2: Which Mamba architecture absorbs distillation best?
Six branches with different internal mechanisms:
- **A** (VMamba+LSTM, 0.97M): has ViT-like encoder + LSTM → most architecturally similar to teacher → **expected best absorber**
- **B/B+** (MambaVision, 2.6M/2.6M): has explicit attention-like mechanisms → easier alignment
- **C** (CNN+Mamba-3, 2.4M): CNN encoder may align naturally with ViT features
- **D** (STH-Mamba, 2.6M): spatiotemporal hybrid
- **E** (DecisionMamba, 2.2M): pure SSM, least similar to teacher → most capacity-constrained for feature alignment

**Prediction**: A > B/B+ > C > D > E (similarity to teacher architecture predicts transfer ease).

### Q3: Distilled Mamba vs pure BC Mamba in simulation?
**Prediction**: Distilled models should show:
- Lower validation loss (already seen in MOHAWK/X-Distill)
- Lower collision rate in Flightmare simulation
- Better generalization to unseen obstacle configurations

### Q4: Novelty claim — first cross-architecture KD in robot control?
- X-Distill does ViT→CNN for manipulation (closest, but not Mamba)
- CADiT does KD for drone depth estimation (not end-to-end control)
- No existing work does ViT+LSTM→Mamba distillation for quadrotor control

**Verdict**: ✅ Genuinely novel contribution if successful.

---

## 4. Implementation Plan

### Files
| File | Purpose | Status |
|------|---------|--------|
| `training/train_distill.py` | Main distillation training script | ⏳ TODO |
| `experiments/distillation/` | Experiment outputs & configs | ✅ Dir exists |
| `experiments/distillation/PROTOCOL.md` | Experiment protocol | ✅ Exists, needs update |
| `literature/survey.md` | Full literature survey | ✅ Created |
| `results/DISTILLATION_REPORT.md` | This report | ✅ Creating |

### Training Pipeline
```
1. Load teacher: ViT+LSTM from checkpoint (eval, freeze)
2. Load student: Mamba branch (random init or pretrained BC weights)
3. For each batch:
   a. Forward teacher → f_teacher, v_teacher
   b. Forward student → f_student, v_student
   c. Compute L = α·MSE(f) + β·MSE(v_t, v_s) + γ·MSE(v_s, v_gt)
   d. Backprop through student only
4. Validate on held-out trajectories
5. Save best checkpoint
6. Test in Flightmare simulation
```

### Success Criteria

**⚠ Reference: Teacher vs BC val_loss is misleading**
Teacher = 0.027 (val_loss) vs BC = 0.016-0.023 (val_loss), but teacher flies 7m/s.
**val_loss is NOT the ground truth for flight quality. Simulation is.**

| Priority | Metric | Target | How to Measure |
|----------|--------|--------|----------------|
| P0 | Collision rate | Lower than BC baseline | Flightmare simulation (≥10 runs) |
| P0 | Flight quality | Smooth trajectory | Visual inspection in Unity |
| P1 | Completion time | Not significantly slower | Simulation test |
| P2 | Val loss | Monitor for divergence/collapse only | Compare to BC at same epoch |
| P2 | Architecture ranking | Identify best absorber | Compare delta over BC per branch |

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Feature dimensions mismatch | Use projection head (linear) to align dims |
| Teacher too different from student | Start with Branch A (has LSTM = most similar) |
| Distillation collapses to teacher mean | Always include GT loss (γ > 0) |
| No simulation improvement | Check proxy metric correlations first |
| Training too slow | Use mixed precision (FP16), gradient accumulation |

---

## 6. References

1. MOHAWK — Bick et al., NeurIPS 2024. Transformers to SSMs: Distilling Quadratic Knowledge.
2. CAB — Wang et al., 2025. Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge.
3. X-Distill — Shao et al., ICLR 2026. Cross-Architecture Vision Distillation for Visuomotor Learning.
4. FASD — 2024. Boosting LiDAR Detection via Cross-Model KD (Transformer→Mamba).
5. DLRMamba — 2026. Distilling Low-Rank Mamba for Edge Deployment.
6. EdgeNavMamba — 2025. Mamba + KD + RL for Edge Navigation.
7. CADiT — 2024. Distilled Depth for Nano-Drones with Channel-Aware Distillation.
8. KD-Mamba — Cheng et al., 2025. SSM with KD for Trajectory Prediction.

See `literature/survey.md` for full summaries of each paper.
