"""
Branch B+: MambaVision Visual Encoder (copied from Branch B)
Hybrid Mamba-Transformer Vision Backbone (NVIDIA)
For end-to-end drone obstacle avoidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaVisionBlock(nn.Module):
    """MambaVision hybrid block (CNN + SSM)"""
    def __init__(self, dim, d_state=16, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        
        # SSM path
        self.ssm = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        # CNN path (local features)
        self.conv_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        
        # SSM path
        x_norm = self.norm1(x)
        ssm_out = self.ssm(x_norm)
        
        # CNN path
        x_conv = rearrange(x_norm, 'b h w c -> b c h w')
        conv_out = self.conv_path(x_conv)
        conv_out = rearrange(conv_out, 'b c h w -> b h w c')
        
        # Fusion
        x = x + ssm_out + conv_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MambaVisionStage(nn.Module):
    """MambaVision stage (multiple blocks)"""
    def __init__(self, in_dim, out_dim, depth, d_state=16, dropout=0.1):
        super().__init__()
        
        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.downsample = nn.Identity()
            
        self.blocks = nn.ModuleList([
            MambaVisionBlock(out_dim, d_state, dropout=dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        """x: (B, C, H, W)"""
        x = self.downsample(x)
        x = rearrange(x, 'b c h w -> b h w c')
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class MambaVisionEncoder(nn.Module):
    """
    MambaVision Visual Encoder (trimmed version, ~2.5M params)
    
    Architecture:
    - Stem: Conv2d(1→64, k=7, s=4)
    - Stage 1: MambaVisionBlock × 4 (embed_dim=96)
    - Stage 2: MambaVisionBlock × 4 (embed_dim=192)
    - Stage 3: MambaVisionBlock × 4 (embed_dim=384)
    - Output: 512-dim features
    """
    
    def __init__(
        self,
        in_channels=1,
        stem_dim=48,
        stage_dims=(64, 128, 192),
        depths=(2, 2, 2),
        d_state=12,
        dropout=0.1,
        output_dim=512
    ):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(stem_dim),
            nn.GELU()
        )
        
        # Stages
        self.stage1 = MambaVisionStage(stem_dim, stage_dims[0], depths[0], d_state, dropout)
        self.stage2 = MambaVisionStage(stage_dims[0], stage_dims[1], depths[1], d_state, dropout)
        self.stage3 = MambaVisionStage(stage_dims[1], stage_dims[2], depths[2], d_state, dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stage_dims[2], output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        """x: (B, 1, 60, 90)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.output_proj(x)
        return x
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_mambavision_encoder(config):
    return MambaVisionEncoder(
        in_channels=config.get('in_channels', 1),
        stem_dim=config.get('stem_dim', 64),
        stage_dims=config.get('stage_dims', (96, 192, 384)),
        depths=config.get('depths', (4, 4, 4)),
        d_state=config.get('d_state', 16),
        dropout=config.get('dropout', 0.1),
        output_dim=config.get('output_dim', 512)
    )
