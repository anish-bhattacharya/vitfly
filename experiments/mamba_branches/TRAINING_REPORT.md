# Mamba Branch Training Report (Updated)

## Training Configuration
- **Framework**: PyTorch with FP16 mixed precision
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.01)
- **Scheduler**: Cosine annealing with warmup
- **Batch Size**: 32
- **Epochs**: 100 (all branches)
- **Data**: 150 samples (120 train, 30 val) - sample-level split

## Results Summary

| Branch | Model | Parameters | Train Loss | Val Loss | Best Val | Overfitting |
|--------|-------|------------|------------|----------|----------|-------------|
| B | MambaVision+SSM | 2.61M | 1.79→0.0004 | 3.72→0.0071 | 0.0069 | ⚠️ 16x |
| C | CNN+Mamba3 | 2.14M | 0.53→0.0011 | 0.40→0.0075 | 0.0057 | ✓ 7x |
| D | STH-Mamba | 2.76M | 0.27→0.0002 | 0.17→0.0057 | 0.0054 | ⚠️ 25x |
| E | DecisionMamba | 1.36M | 11.4→0.0052 | 9.06→0.0057 | 0.0055 | ✓ 1x |

## Issues Fixed

### 1. Target Variable Bug (CRITICAL)
**Before**: `target = [desired_vels[idx]] * 3` (repeated scalar)
**After**: `target = velocity.clone()` (correct 3D velocity)

This was causing the model to learn a trivial mapping instead of proper velocity prediction.

### 2. Empty Validation Set (FIXED)
- Original: trajectory-level split with 3 trajs → 0 validation samples
- Fixed: sample-level split → 30 validation samples

### 3. Branch E 10 Epochs (FIXED)
- Original: trained with wrong script (10 epochs)
- Fixed: retrained with 100 epochs

## Overfitting Analysis

- **Branch E**: Best generalization (gap=1.1x) - smallest model
- **Branch C**: Good generalization (gap=7x)
- **Branch B/D**: Some overfitting (16x, 25x) - larger models

With more data, overfitting would reduce. Current sample data is too small (150 samples).

## Conclusion

1. ✅ Training converges properly now (correct data pipeline)
2. ⚠️ Some overfitting observed in larger models (B, D) - needs more data
3. ✓ Branch E generalizes best despite being retrained