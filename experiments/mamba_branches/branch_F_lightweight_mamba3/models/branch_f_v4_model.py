"""
Branch F v4: MobileNetV3 Encoder (Branch C) + Mamba-3 Head (Branch F)

Combines Branch C's MobileNetV3-style CNN encoder (0.41M params) with
Branch F's Mamba-3 SSM temporal head (d_state=64, hidden=512) for
end-to-end drone obstacle avoidance. Total: ~2.46M params.

Architecture:
  Depth (60x90)
      |
  MobileNetV3 CNN Encoder (Branch C, 0.41M)
  |- Stem: Conv2d(1->32, k=3, s=2) + BN + GELU
  |- InvertedResidual blocks: 32->64->128->256
  |- AdaptiveAvgPool -> Linear(256->512)
      |
  [features (512), vel (3), quat (4)] -- concat --
      |
  Mamba-3 SSM Head (Branch F, d_state=64, hidden=512, 2.04M)
  |- Input projection (519 -> 512)
  |- Mamba3Block x 2 (trapezoidal discretization, RoPE, QK-norm)
  |- RMSNorm output norm
      |
  SpectralNorm Linear(512 -> 3)
      |
  Velocity command (vx, vy, vz)

Purpose: Test if encoder capacity is the bottleneck. Branch F lightweight
CNN (0.5M) led to 7 crashes, while Branch C's MobileNetV3 (0.41M) with
smaller Mamba-3 had 3 crashes. This hybrid pairs the stronger encoder
with Branch F's larger Mamba-3 SSM head.
"""

import os
import sys

_model_dir_c = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'branch_C_cnn_mamba3', 'models')
)
_model_dir_f = os.path.abspath(
    os.path.join(os.path.dirname(__file__))
)
for _p in [_model_dir_c, _model_dir_f]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from cnn_encoder import CNNEncoder, create_cnn_encoder
from mamba3_head import Mamba3Head, create_mamba3_head


class RefineInputs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear',
                                 align_corners=False)
        return X


class BranchFV4Model(nn.Module):
    """
    Branch F v4: MobileNetV3 Encoder (Branch C) + Mamba-3 Head (Branch F).

    Combines Branch C's MobileNetV3-style CNN encoder with Branch F's
    Mamba-3 SSM head (d_state=64, hidden=512). Total: ~2.46M params.
    """

    def __init__(
        self,
        cnn_config=None,
        mamba3_d_state=64,
        mamba3_hidden=512,
        mamba3_layers=2,
        mamba3_headdim=32,
        mamba3_chunk_size=32,
        dropout=0.1,
    ):
        super().__init__()

        if cnn_config is None:
            cnn_config = {
                'in_channels': 1,
                'stem_dim': 32,
                'stage_dims': (32, 64, 128, 256),
                'output_dim': 512,
                'dropout': dropout,
            }

        self.cnn_encoder = create_cnn_encoder(cnn_config)
        cnn_output_dim = cnn_config.get('output_dim', 512)

        self.refine = RefineInputs()

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
        X = self.refine(X)

        depth_feat = self.cnn_encoder(X[0])

        metadata = torch.cat((X[1] * 0.1, X[2]), dim=1).float()
        x = torch.cat((depth_feat, metadata), dim=1)

        x, hidden = self.mamba3_head(x)

        if x.dim() == 3:
            x = x[:, -1, :]

        x = self.fc_out(x)

        return x, hidden

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def print_param_breakdown(self):
        components = {
            'CNN Encoder': self.cnn_encoder,
            'Mamba-3 Head': self.mamba3_head,
            'FC Out': self.fc_out,
        }
        total = 0
        model_total = sum(p.numel() for p in self.parameters())
        print(f"{'Component':<16} {'Params':>10}")
        print('-' * 28)
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            pct = n / model_total * 100 if model_total > 0 else 0
            print(f"{name:<16} {n:>10,}  ({pct:.1f}%)")
        print('-' * 28)
        print(f"{'TOTAL':<16} {total:>10,}")
        return total


def create_branch_f_v4_model(config):
    return BranchFV4Model(
        cnn_config=config.get('cnn_config'),
        mamba3_d_state=config.get('mamba3_d_state', 64),
        mamba3_hidden=config.get('mamba3_hidden', 512),
        mamba3_layers=config.get('mamba3_layers', 2),
        mamba3_headdim=config.get('mamba3_headdim', 32),
        mamba3_chunk_size=config.get('mamba3_chunk_size', 32),
        dropout=config.get('dropout', 0.1),
    )


if __name__ == '__main__':
    model = BranchFV4Model()
    params = model.get_parameter_count()
    print(f"Branch F v4 Total Parameters: {params:,} ({params/1e6:.2f}M)\n")
    model.print_param_breakdown()

    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4),
    ]

    with torch.no_grad():
        output, hidden = model(X)

    print(f"\nInput shapes:")
    print(f"  depth:  {X[0].shape}")
    print(f"  vel:    {X[1].shape}")
    print(f"  quat:   {X[2].shape}")
    print(f"Output:   {output.shape}")
    print(f"Hidden:   {hidden}")
