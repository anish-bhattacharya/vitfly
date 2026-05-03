# Cross-Architecture Knowledge Distillation: ViT+LSTM → Mamba

## Experiment Protocol

### Hypothesis
Knowledge from a pretrained ViT+LSTM teacher (upstream best model, 7m/s real flight)
can be transferred to Mamba-based student architectures via distillation,
outperforming behavior cloning from scratch.

### Literature Basis
- MOHAWK (NeurIPS 2024): Three-phase Transformer→Mamba-2 distillation
- CAB (2025): Attention bridge for cross-architecture alignment
- X-Distill (ICLR 2026): ViT→CNN distillation for robotics
- TransMamba (2025): Vision Mamba cross-architecture transfer

### Shared Infrastructure (inherited from `mambatest`)
- Model definitions under `experiments/mamba_branches/branch_{name}/models/`
- Training pipeline under `training/` (`train_mamba_optimized.py`, `lazy_dataloading.py`)
- Data under `training/datasets/data_full/`
- Simulation pipeline under `results/`

### Changes on This Branch
- `experiments/distillation/` — experiment code (this directory)
- `training/train_distill.py` — distillation training script (NEW, not modifying existing)
- Teacher model: `models/ViTLSTM_model.pth` (14MB, from upstream)
