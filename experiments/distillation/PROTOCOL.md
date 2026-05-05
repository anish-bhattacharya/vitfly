# Cross-Architecture Knowledge Distillation: ViT+LSTM → Mamba

## Experiment Protocol (Updated 2026-05-04)

### Hypothesis
Knowledge from a pretrained ViT+LSTM teacher (upstream best model, 7m/s real flight)
can be transferred to Mamba-based student architectures via multi-stage distillation,
outperforming behavior cloning from scratch.

### Literature Basis (Detailed)

| Paper | Venue | Key Insight for Our Work |
|-------|-------|-------------------------|
| **MOHAWK** | NeurIPS 2024 | Three-phase (matrix→hidden→output) is **required** for cross-architecture; naive output-only KD fails |
| **CAB** | 2025 | Attention bridge: map Transformer Q,K ↔ Mamba B,C via MLP for token-level alignment |
| **X-Distill** | ICLR 2026 | ViT encoder → compact student + robotics fine-tune; beats 3D encoders |
| **FASD** | 2024 | Heterogeneous Transformer→Mamba with adapter-based feature alignment + Span-KD |
| **DLRMamba** | 2026 | Multi-level distillation: SVD alignment + hidden state + feature reconstruction |
| **EdgeNavMamba** | 2025 | Mamba+KD on edge (67% smaller, 73% less energy, same accuracy) |
| **CADiT** | 2024 | Channel-attention KD for nano-drone depth estimation |
| **KD-Mamba** | 2025 | Mamba + KD for trajectory prediction |

See `literature/survey.md` for full paper summaries.

### Design Evolution (Based on Literature)

**Previous design** (2-stage):
```
Phase 1: feature alignment
Phase 2: output distillation + GT
```

**Updated design** (3-phase, MOHAWK-inspired):
```
Phase 1: Encoder feature alignment (MSE of encoder outputs)
         → X-Distill style: align student encoder features to teacher ViT features

Phase 2: Output distillation (temperature-scaled soft targets)
         → MOHAWK Stage 3: match student velocity to teacher velocity

Phase 3: Joint training with GT grounding
         L = α·L_feat + β·L_distill + γ·L_gt
```

### Ablation Experiments

| ID | α | β | γ | Tests |
|----|---|---|---|-------|
| C0 | 0 | 0 | 1 | Pure BC baseline |
| C1 | 1 | 0 | 1 | Feature alignment only |
| C2 | 0 | 1 | 1 | Output KD only (tests MOHAWK's claim that this fails) |
| C3 | 1 | 1 | 1 | Full multi-stage |

### 6 Architectures to Compare

| Branch | Model | Params | BC Val Loss | BC Checkpoint |
|--------|-------|--------|-------------|---------------|
| A | VMamba+LSTM | 974K | 0.0161 (ep27) | ✅ exists |
| B | MambaVision+SSM | 2.61M | 0.0205 (ep4) | ✅ exists |
| Bplus | MambaVision+Mamba-3 | 2.55M | 0.0231 (ep1) | ✅ exists |
| C | CNN+Mamba-3 | 2.41M | 0.0221 (ep1) | ✅ exists |
| D | STH-Mamba | 2.60M | 0.0173 (ep1) | ✅ exists |
| E | DecisionMamba | 2.19M | 0.0186 (ep1) | ✅ exists |

**Predicted absorption ranking**: A > B/B+ > C > D > E
- **A** (VMamba+LSTM): most similar to teacher (ViT encoder + LSTM) → easiest feature alignment
- **B/B+** (MambaVision): built-in attention mechanism → likely good cross-architecture transfer
- **C** (CNN+Mamba-3): simple CNN features → moderate alignment difficulty
- **D** (STH-Mamba): spatiotemporal hybrid → harder alignment
- **E** (DecisionMamba): pure SSM, least similar → hardest, but may benefit most from KD

### Success Criteria
1. **Primary: Simulation performance** (Flightmare / Unity)
   - Collision rate (10+ runs per model)
   - Completion time (not significantly slower than BC)
   - Visual inspection of flight trajectory quality
2. **Secondary: Val loss** — used as a diagnostic, NOT the primary decision metric.
   - Teacher has higher val_loss (0.027) than BC models (0.016-0.023) but flies better at 7m/s
   - A distillation student may also show higher val_loss while flying better
   - Monitor val_loss for divergence/collapse, not for ranking
3. At least one branch shows significant improvement in simulation

### Phase 1 Results (2026-05-04, all 6 branches distlled)

**Training**: 50 epochs each, C3 config (α=β=γ=1), 42K images, from scratch
**Verification**: All branches converged without NaN/divergence ✅

| Branch | Best score | GT loss | Distill gap | vs BC (gt) |
|--------|-----------|---------|-------------|------------|
| **D** 🏆 | **0.0258** | 0.0173 | 0.0171 | ±0.0000 |
| A | 0.0266 | 0.0184 | 0.0165 | +0.0023 |
| E | 0.0274 | 0.0188 | 0.0172 | +0.0002 |
| C | 0.0276 | 0.0188 | 0.0176 | -0.0033 |
| Bplus | 0.0283 | 0.0196 | 0.0173 | -0.0035 |
| B | 0.0284 | 0.0201 | 0.0165 | -0.0004 |

**Pending**: Flightmare simulation to verify collision rate improvement

### Shared Infrastructure (inherited from `mambatest`)
- Model definitions under `experiments/mamba_branches/branch_{name}/models/`
- Training pipeline under `training/` (`train_mamba_optimized.py`, `lazy_dataloading.py`)
- Data under `training/datasets/data_full/`
- Simulation pipeline under `results/`

### Changes on This Branch
- `experiments/distillation/` — experiment code (this directory)
- `training/train_distill.py` — distillation training script (NEW)
- `literature/` — paper summaries and survey
- `results/DISTILLATION_REPORT.md` — comprehensive report
- Teacher model: `models/ViTLSTM_model.pth` (14MB, from upstream)
