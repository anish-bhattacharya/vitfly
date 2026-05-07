"""
Branch F: Lightweight CNN Encoder

Efficient convolutional encoder for drone depth maps.
Target: ~0.5M parameters.

Architecture:
- Stem: Conv2d(1->32, k=7, s=4) + BN + GELU
- Stage 1: Conv2d(32->64, k=3, s=2) + 2x Conv2d(64->64, k=3, s=1)
- Stage 2: Conv2d(64->128, k=3, s=2) + 2x Conv2d(128->128, k=3, s=1)
- Head: AdaptiveAvgPool -> FC(128->256)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + GELU block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightweightCNNEncoder(nn.Module):
    """
    Lightweight CNN encoder for depth map feature extraction.

    Parameter breakdown (target ~0.5M total):
      Stem:             1,632
      Stage 1:         92,544
      Stage 2:        369,408
      Head:            33,024
      -----------------------
      Total:          496,608 (~0.50M)

    Input:  (B, 1, 60, 90) depth map
    Output: (B, 256) feature vector
    """

    def __init__(self, in_channels=1, output_dim=256):
        super().__init__()

        # Stem: aggressive downsampling to reduce spatial size early
        # Conv2d(1->32, k=7, s=4): 60x90 -> 15x23 (after padding)
        self.stem = ConvBlock(in_channels, 32, kernel_size=7, stride=4, padding=3)

        # Stage 1: 32 -> 64 channels, stride 2 downsampling
        # Input @ 15x23, output @ 8x12
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # Stage 2: 64 -> 128 channels, stride 2 downsampling
        # Input @ 8x12, output @ 4x6
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )

        # Head: global average pooling -> 256-dim feature vector
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x)
        return x

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def print_param_breakdown(self):
        """Print per-component parameter counts."""
        components = {'stem': self.stem, 'stage1': self.stage1,
                      'stage2': self.stage2, 'head': self.head}
        total = 0
        print(f"{'Component':<12} {'Params':>10}")
        print('-' * 22)
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            print(f"{name:<12} {n:>10,}")
        print('-' * 22)
        print(f"{'Total':<12} {total:>10,}")
        return total


def create_lightweight_cnn_encoder(config):
    """Factory function matching the pattern used by other branches."""
    return LightweightCNNEncoder(
        in_channels=config.get('in_channels', 1),
        output_dim=config.get('output_dim', 256),
    )


if __name__ == '__main__':
    encoder = LightweightCNNEncoder()
    params = encoder.get_parameter_count()
    print(f"Lightweight CNN Encoder Parameters: {params:,} ({params/1e6:.2f}M)")
    encoder.print_param_breakdown()

    x = torch.randn(1, 1, 60, 90)
    with torch.no_grad():
        output = encoder(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {output.shape}")
