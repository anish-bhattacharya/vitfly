# Branch B+: MambaVision + Mamba-3 SSM

**Status**: ✅ Implementation Complete  
**Parameters**: ~11.8M  
**Architecture**: MambaVision visual encoder + Real Mamba-3 SSM temporal head

## Overview

Branch B+ combines the MambaVision hybrid visual encoder from Branch B with a **real Mamba-3 SSM implementation** adapted from the [mamba3-minimal](https://github.com/state-spaces/mamba) reference. This replaces the simplified SSM head in Branch B with a production-grade Mamba-3 architecture.

## Architecture

```
Input (depth map 60×90)
    ↓
MambaVision Encoder (~11.2M params)
├─ Stem: Conv2d(1→48, k=7, s=4)
├─ Stage 1: MambaVisionBlock × 2 (64 channels)
├─ Stage 2: MambaVisionBlock × 2 (128 channels)
└─ Stage 3: MambaVisionBlock × 2 (192 channels)
    ↓
Feature Vector (512-dim)
    ↓
Concatenate with metadata (vel + quat)
    ↓
Mamba-3 SSM Head (~0.6M params)
├─ Input projection (519 → 256)
├─ Mamba3Block × 2
│  ├─ Trapezoidal discretization (α, β, γ)
│  ├─ Data-dependent RoPE
│  ├─ QK-Normalization on B, C
│  ├─ Learnable BC bias
│  └─ Two-SSD decomposition
└─ Output projection (256 → 3)
    ↓
Velocity command (vx, vy, vz)
```

## Key Innovations (Mamba-3)

### 1. Trapezoidal Discretization
- **2nd-order accurate** state update vs. Euler (1st-order)
- Recurrence: `h_t = α_t h_{t-1} + β_t B_{t-1}x_{t-1} + γ_t B_t x_t`
- Requires two-SSD decomposition for parallel training

### 2. Data-Dependent RoPE
- Complex-valued SSM via rotary position embeddings
- Angles computed from input: `θ_t = Δ_t * θ_proj(x_t)`
- Enables state-tracking (100% parity task accuracy)

### 3. QK-Normalization
- RMSNorm applied to B, C projections
- Improves training stability
- Mirrors QK-Norm in Transformers

### 4. Learnable BC Bias
- Head-specific, channel-wise bias
- Initialized to ones
- Data-independent, trainable

### 5. No Short Convolution
- Trapezoidal rule + bias eliminates need for Conv1d
- Simpler architecture, fewer parameters

## Implementation Details

### Two-SSD Decomposition
The trapezoidal recurrence introduces a strict cross-boundary dependency (β term depends on previous timestep). Standard PyTorch chunking breaks here. Solution:

1. **γ SSD** (current timestep): `ssd(x * γ, dA, B, C)`
2. **β SSD** (previous timestep): `ssd(x_prev * β, dA, B_prev, C)`

Pre-shift B and x at the **global sequence level** before chunking to handle cross-chunk boundaries naturally.

### Sequence Length Handling
Mamba-3 requires sequence length divisible by `chunk_size` (default: 32). The implementation automatically pads sequences during forward pass and unpad before output.

## Files

```
branch_Bplus_mambavision_mamba3/
├── models/
│   ├── __init__.py              # Module exports
│   ├── mambavision_encoder.py   # MambaVision visual encoder (from Branch B)
│   ├── mamba3_head.py           # Real Mamba-3 SSM implementation
│   └── bplus_model.py           # Complete model integration
└── README.md                    # This file
```

## Usage

### Standalone Testing

```python
import torch
from models import create_bplus_model

model = create_bplus_model({})
print(f"Parameters: {model.get_parameter_count():,}")

X = [
    torch.randn(1, 1, 60, 90),  # depth map
    torch.randn(1, 3),           # velocity
    torch.randn(1, 4)            # quaternion
]

output, hidden = model(X)
print(f"Output shape: {output.shape}")  # torch.Size([1, 3])
```

### Training Integration

```bash
cd /root/vitfly/training
python train_mamba_optimized.py \
    --branches Bplus \
    --epochs 100 \
    --batch_size 32 \
    --data_dir datasets/data_full
```

## Configuration

Default hyperparameters:

```python
{
    'mambavision_config': {
        'in_channels': 1,
        'stem_dim': 48,
        'stage_dims': (64, 128, 192),
        'depths': (2, 2, 2),
        'd_state': 12,
        'dropout': 0.1,
        'output_dim': 512
    },
    'mamba3_d_state': 64,      # SSM state dimension
    'mamba3_hidden': 256,       # Hidden dimension
    'mamba3_layers': 2,         # Number of Mamba-3 blocks
    'mamba3_headdim': 32,       # Head dimension
    'mamba3_chunk_size': 32,    # Chunk size for SSD
    'dropout': 0.1
}
```

## Expected Performance

Based on Branch B results and Mamba-3 improvements:

- **Convergence**: Expected to match or exceed Branch B (val loss ~0.000001)
- **Training stability**: Improved via QK-Normalization
- **Temporal modeling**: Better long-range dependencies via trapezoidal discretization
- **State tracking**: Enhanced via data-dependent RoPE

## References

1. **Mamba-3 Paper**: "Mamba-3: Improved Sequence Modeling Using State Space Principles" (ICLR 2026, under review)
2. **mamba3-minimal**: https://github.com/state-spaces/mamba (reference implementation)
3. **MambaVision**: NVIDIA's hybrid Mamba-Transformer vision backbone
4. **Original ViTFly**: "Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance" (ICRA 2025)

## Comparison with Branch B

| Feature | Branch B | Branch B+ |
|---------|----------|-----------|
| Visual Encoder | MambaVision | MambaVision (same) |
| Temporal Head | Simplified SSM | Real Mamba-3 SSM |
| Discretization | Euler (1st-order) | Trapezoidal (2nd-order) |
| Position Encoding | None | Data-dependent RoPE |
| Normalization | Basic | QK-Normalization |
| Parameters | ~3.3M | ~11.8M |
| State Tracking | Limited | Enhanced |

## Next Steps

1. Train on full dataset (200 trajectories, 100 epochs)
2. Compare validation loss with Branch B
3. Evaluate in simulation (Flightmare)
4. Ablation studies:
   - Trapezoidal vs. Euler discretization
   - With/without data-dependent RoPE
   - Different d_state values (32, 64, 128)

## Notes

- The Mamba-3 implementation is adapted from the official reference but simplified for this specific use case
- Chunk size must divide sequence length; automatic padding is applied
- The model is hardware-agnostic (CUDA, MPS, CPU)
- For production deployment, consider the official Mamba-3 CUDA kernels for maximum performance
