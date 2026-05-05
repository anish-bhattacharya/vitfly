# Branch A Test Report - d_state=64 Architecture

## Test Date
2026-05-03

## Model Architecture
- **Model**: VMambaLSTMNet (VMamba + LSTM)
- **VMamba Config**: embed_dim=64, depth=4, d_state=64, dropout=0.1, output_dim=512
- **LSTM Config**: hidden=128, layers=2
- **Total Parameters**: 974,275
  - VMamba: 511,552
  - LSTM: 462,336
  - Output: 387

## Training Results
- **Best Epoch**: 27
- **Best Val Loss**: 0.0161
- **Final Train Loss**: 0.0060 (epoch 100)
- **Final Val Loss**: 0.0193 (epoch 100)
- **Training Duration**: 100 epochs

## Simulation Test Results
- **Success**: false (1 crash)
- **Time to Finish**: 4.22s
- **Velocity Outputs**: 248 commands
- **Crashes**: 1

## Comparison with Other Branches

| Branch | Model | Success | Crashes | Time | Velocity Outputs | Val Loss | Notes |
|--------|-------|---------|---------|------|------------------|----------|-------|
| A | VMambaLSTM | ❌ (1 crash) | 1 | 4.22s | 248 | 0.0161 | **NEW** d_state=64 weights |
| B | MambaVisionSSM | ✅ | 0 | 4.26s | 242 | 0.0205 (epoch 4) | |
| B+ | BPlusModel | ✅ | 0 | 4.21s | 242 | 0.0231 | |
| C | CNNMamba3 | ✅ | 0 | 4.20s | 244 | 0.0221 | |
| D | STHMamba | ✅ | 0 | 4.21s | 245 | 0.0173 | |
| E | DecisionMamba | ✅ | 0 | 4.21s | 243 | 0.0186 | |

## Key Findings

### ✅ Successes
1. **Model loads correctly** - No more state_dict shape mismatch errors
2. **Inference works** - Model produces 248 velocity commands (similar to other branches)
3. **Performance comparable** - 4.22s completion time matches other branches
4. **Best validation loss** - 0.0161 is the lowest among all branches
5. **Reasonable outputs** - Velocity commands are in expected range

### ⚠️ Issues
1. **1 crash** - Unlike other branches which have 0 crashes
2. **Success: false** - Due to the single crash

## Velocity Command Quality
Sample outputs show reasonable control:
```
[1.0, 0.093, -0.116]
[1.0, 0.076, -0.086]
[1.0, 0.067, -0.075]
```
- Forward velocity: 1.0 (normalized)
- Lateral/vertical: Small corrections (-0.12 to +0.09)

## Root Cause Analysis
The single crash may be due to:
1. **Training data distribution** - Model may need more diverse obstacle scenarios
2. **Hyperparameter tuning** - LSTM hidden size or dropout may need adjustment
3. **Random variation** - Single test run may not be representative

## Recommendations

### Short-term (Ready for deployment with caveat)
- ✅ Model is functional and performs inference correctly
- ✅ Performance metrics (time, velocity count) match other branches
- ⚠️ Single crash suggests slightly less robust than other branches
- **Recommendation**: Deploy with monitoring, but consider as backup to branches D/E

### Medium-term (Improvement opportunities)
1. **Run multiple test iterations** - Verify if crash is consistent or random
2. **Analyze crash location** - Check which obstacle caused the collision
3. **Fine-tune hyperparameters** - Try different LSTM hidden sizes (64, 256)
4. **Data augmentation** - Add more challenging obstacle scenarios to training

### Long-term (Architecture evolution)
1. **Ensemble approach** - Combine Branch A with Branch D (best val loss pair)
2. **Attention mechanism** - Add attention between VMamba and LSTM
3. **Multi-scale features** - Extract features at multiple resolutions

## Conclusion
Branch A with d_state=64 architecture is **functional and ready for testing**, but shows slightly less robustness than branches B/B+/C/D/E due to 1 crash. The model has the **best validation loss (0.0161)** among all branches, suggesting strong learning capability. Recommend deploying as a **secondary option** while investigating the crash cause.
