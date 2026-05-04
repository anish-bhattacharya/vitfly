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

### Open Questions
- Does feature alignment (L_feat) actually help in this visual domain? MOHAWK showed it's essential for NLP, but vision may differ
- Which branch architecture benefits most from distillation? (Prediction: A > B/B+ > C > D > E)
- Can distillation overcome the teacher's higher val_loss and actually improve simulation performance?
- Are different α,β,γ needed per branch, or is a single setting sufficient?
