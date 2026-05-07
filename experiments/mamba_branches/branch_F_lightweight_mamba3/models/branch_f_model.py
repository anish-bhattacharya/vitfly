"""
Branch F: Lightweight CNN + Mamba-3 SSM

Combines a lightweight convolutional encoder (~0.5M) with the proven
Mamba-3 SSM temporal head (~0.6M) for end-to-end drone obstacle avoidance.

Total parameters: ~1.1M

Architecture:
  Depth (60x90)
      |
  Lightweight CNN Encoder (0.5M)
      |
  [features (256), vel (3), quat (4)]  -- concat --
      |
  Mamba-3 SSM Head (0.6M)
  ├─ Input projection (263 -> 256)
  ├─ Mamba3Block x 2
  │  ├─ Trapezoidal discretization
  │  ├─ Data-dependent RoPE
  │  ├─ QK-Normalization
  │  └─ Two-SSD decomposition
  └─ Output norm
      |
  Linear(256 -> 3)
      |
  Velocity command (vx, vy, vz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from lightweight_cnn_encoder import LightweightCNNEncoder, create_lightweight_cnn_encoder
from mamba3_head import Mamba3Head, create_mamba3_head


class RefineInputs(nn.Module):
    """Normalize and prepare inputs for the model."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Handle missing quaternion (default to identity)
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        # Resize depth to expected dimensions
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear',
                                 align_corners=False)
        return X


class BranchFModel(nn.Module):
    """
    Lightweight CNN + Mamba-3 hybrid for drone obstacle avoidance.

    Combines:
    - LightweightCNNEncoder: efficient depth feature extraction (~0.5M)
    - Mamba3Head: temporal SSM modeling with trapezoidal discretization (~0.6M)
    - Output: velocity command (vx, vy, vz) via spectral norm linear layer

    Parameters by component:
      LightweightCNNEncoder:         496,608  (~0.50M)
      Mamba3Head (input_dim=263):   597,800  (~0.60M)
      fc_out:                           771  (~0.00M)
      ------------------------------------------
      Total:                      1,095,179  (~1.10M)
    """

    def __init__(
        self,
        cnn_encoder_config=None,
        mamba3_d_state=64,
        mamba3_hidden=256,
        mamba3_layers=2,
        mamba3_headdim=32,
        mamba3_chunk_size=32,
        dropout=0.1
    ):
        super().__init__()

        if cnn_encoder_config is None:
            cnn_encoder_config = {
                'in_channels': 1,
                'output_dim': 256,
            }

        # Depth feature extractor
        self.cnn_encoder = create_lightweight_cnn_encoder(cnn_encoder_config)
        cnn_output_dim = cnn_encoder_config.get('output_dim', 256)

        # Input refinement
        self.refine = RefineInputs()

        # Mamba-3 SSM head
        # Input = cnn_features(256) + velocity(3) + quaternion(4) = 263
        mamba3_input_dim = cnn_output_dim + 3 + 4

        self.mamba3_head = create_mamba3_head({
            'input_dim': mamba3_input_dim,
            'd_state': mamba3_d_state,
            'hidden_dim': mamba3_hidden,
            'num_layers': mamba3_layers,
            'headdim': mamba3_headdim,
            'chunk_size': mamba3_chunk_size,
            'dropout': dropout,
        })

        # Output projection to velocity command
        self.fc_out = spectral_norm(nn.Linear(mamba3_hidden, 3))

    def forward(self, X):
        """
        Args:
            X: list [depth_map, velocity, quaternion]
                depth_map:   (B, 1, H, W)
                velocity:    (B, 3) or None
                quaternion:  (B, 4) or None

        Returns:
            output: (B, 3) velocity command (vx, vy, vz)
            hidden: SSM hidden state for temporal continuity
        """
        # Step 1: Refine inputs
        X = self.refine(X)

        # Step 2: Extract depth features via lightweight CNN
        depth_feat = self.cnn_encoder(X[0])  # (B, 256)

        # Step 3: Concatenate with metadata
        metadata = torch.cat((X[1] * 0.1, X[2]), dim=1).float()  # (B, 7)
        x = torch.cat((depth_feat, metadata), dim=1)  # (B, 263)

        # Step 4: Temporal modeling via Mamba-3 SSM head
        x, hidden = self.mamba3_head(x)

        # Step 5: Squeeze sequence dim if present
        if x.dim() == 3:
            x = x[:, -1, :]

        # Step 6: Output velocity command
        x = self.fc_out(x)

        return x, hidden

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def print_param_breakdown(self):
        """Breakdown of parameters by component."""
        components = {
            'CNN Encoder': self.cnn_encoder,
            'Mamba-3 Head': self.mamba3_head,
            'FC Out': self.fc_out,
        }
        total = 0
        print(f"{'Component':<16} {'Params':>10}")
        print('-' * 28)
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            pct = n / (sum(p.numel() for p in self.parameters())) * 100
            print(f"{name:<16} {n:>10,}  ({pct:.1f}%)")
        print('-' * 28)
        print(f"{'TOTAL':<16} {total:>10,}")
        return total


def create_branch_f_model(config):
    """Factory function for Branch F model."""
    return BranchFModel(
        cnn_encoder_config=config.get('cnn_encoder_config'),
        mamba3_d_state=config.get('mamba3_d_state', 64),
        mamba3_hidden=config.get('mamba3_hidden', 256),
        mamba3_layers=config.get('mamba3_layers', 2),
        mamba3_headdim=config.get('mamba3_headdim', 32),
        mamba3_chunk_size=config.get('mamba3_chunk_size', 32),
        dropout=config.get('dropout', 0.1),
    )


if __name__ == '__main__':
    model = BranchFModel()
    params = model.get_parameter_count()
    print(f"Branch F Total Parameters: {params:,} ({params/1e6:.2f}M)\n")
    model.print_param_breakdown()

    # Test forward pass
    X = [
        torch.randn(1, 1, 60, 90),  # depth map
        torch.randn(1, 3),           # velocity (vx, vy, vz)
        torch.randn(1, 4),           # quaternion (w, x, y, z)
    ]

    with torch.no_grad():
        output, hidden = model(X)

    print(f"\nInput shapes:")
    print(f"  depth:  {X[0].shape}")
    print(f"  vel:    {X[1].shape}")
    print(f"  quat:   {X[2].shape}")
    print(f"Output:   {output.shape}")
    print(f"Hidden:   {hidden}")
