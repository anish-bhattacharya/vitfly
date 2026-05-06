# Findings: Cross-Architecture Knowledge Distillation for Drone Obstacle Avoidance

## Current Understanding

### Teacher Quality
- ViT+LSTM teacher (upstream best, 7m/s real flight) has higher val_loss (0.027) than BC-trained Mamba branches (0.016-0.023)
- This is NOT a bug — it reflects the teacher being optimized for high-speed flight (7m/s) while our dataset is collected at lower speeds (~5m/s)
- The teacher's knowledge is about high-agility/aggressive obstacle avoidance, not merely minimizing MSE on the training set
- **Implication**: val_loss alone cannot determine flight quality. Simulation evaluation is the ground truth.

### Why Distillation Might Help
- Cross-architecture distillation (X-Distill, ICLR 2026) showed ViT → CNN students can outperform both teacher and from-scratch training in robotics
- The teacher's soft targets provide regularization + high-speed strategy knowledge
- Student's GT loss anchors it to the task domain while distillation transfers the teacher's visual representations
- Mamba's SSM inductive bias may generalize differently from ViT+LSTM — potentially better in some regimes

### Why The Student Might Surpass the Teacher (Three Mechanisms)
1. **X-Distill mechanism**: Distill encoder, fine-tune temporal head on GT → student combines best of both
2. **Teacher regularization**: Teacher soft labels + GT hard labels → mixed strategy may exceed either alone
3. **Architecture advantage**: Mamba's linear complexity allows different/richer representations per parameter

### Key Experimental Design Decisions (as of 2026-05-04)
- **Two tracks**: from scratch (causal attribution) and from BC checkpoint (practical improvement)
- **Best model selection**: `score = val_loss_gt + 0.5 * val_distill_gap`
- **Primary metric**: Flightmare simulation (collision rate), not val_loss
- **Ablation design**: All C0-C5 use same branch (B) with varying α,β,γ — separate from cross-architecture comparison
- **Model architecture**: Distill config matches BC config exactly (verified: all 6 branches have identical params/keys)

### Phase 1 Results (2026-05-04) — All 6 Branches Distilled

| Branch | Best score | GT loss | Distill gap | vs BC (gt) | Analysis |
|--------|-----------|---------|-------------|------------|----------|
| **D** 🏆 | **0.0258** | 0.0173 | 0.0171 | ±0.0000 | STH-Mamba: strong encoder + independent temporal head |
| A | 0.0266 | 0.0184 | 0.0165 | +0.0023 | VMamba+LSTM: most similar to teacher, small capacity |
| E | 0.0274 | 0.0188 | 0.0172 | +0.0002 | DecisionMamba: pure SSM, stable |
| C | 0.0276 | 0.0188 | 0.0176 | **-0.0033** | CNN+Mamba-3: GT improved |
| Bplus | 0.0283 | 0.0196 | 0.0173 | **-0.0035** | MambaVision+Mamba-3: GT improved most |
| B | 0.0284 | 0.0201 | 0.0165 | -0.0004 | MambaVision+SSM: solid baseline |

**Key findings**:
- Prediction was WRONG (A predicted 1st, actual 2nd; D predicted last, actual 1st)
- Architectural similarity ≠ distillation efficiency. STH-Mamba's encoder/head separation may be key
- 4/6 branches improved GT loss over BC baseline — distillation did NOT hurt performance
- All branches converged to similar distill_gap (~0.0165-0.0176)

### Phase 2 Results (2026-05-05) — Simulation @ 20m

**BC vs Distill @ 5m/s:**
- E (DecisionMamba): 0 crashes at both speeds ✅ **clear winner**
- A/B: 1 crash at 5m/s → **0 crashes at 7m/s** 🎉 (distill transferred teacher's high-speed strategy)
- B+/C/D: 1 crash regardless of speed — systematic policy issue, not speed-dependent
- Only val_loss rank D (STH-Mamba) ≠ simulation rank — val_loss is not predictive

**Updated architecture ranking**: E > A ≈ B > D ≈ C ≈ B+
- E won because it's pure SSM with least architectural similarity to teacher
- Architectural similarity ≠ distillation quality (contradicts initial prediction)

**Test limitation**: All runs at 20m (evaluation_config target:20, not 60). Full course pending.

### Open Questions (Updated)
- ~~Which branch architecture benefits most from distillation?~~ → ✅ **E (DecisionMamba)** in simulation
### MambaFusion Negative Result (2026-05-06)

**Architecture**: MambaVision encoder (1.93M) + CoarseSSM (1.12M) + fusion = 3.19M
**Simulation** (3 seeds, 60m):
- Distill s44 @ 5m/s: **1 crash** 🏆 (matches E Distill)
- Distill other seeds: 4, 4 crashes (high variance)
- @ 7m/s: mixed (s42=4, s43=2, s44=2)
- BC: very high variance (2, 4, 7 crashes)

**Root cause** (from literature):
1. **Mamba training instability** (B2S6, 2025): S6 models provably unstable when scaled. Larger models (3.19M vs 2.19M) amplify this.
2. **Initialization sensitivity** (Mimetic Init, ICLR 2025): SSM init causes up to 16× variance. Our 1-4 crash range matches.
3. **Data-limited regime**: 42K trajectories insufficient for 3.19M model. E (2.19M) at sweet spot.

**Corrected SSM design principle**: Lightweight pure-SSM beats heavy hybrid-SSM in data-limited robot learning.

**Next direction**: E-SSM — replace E's CNN encoder with lightweight SSM encoder (256-dim), keep E's proven temporal. Expected ~2.2M, stable.
