"""
分支 C: CNN 编码器 (MobileNetV3 风格)
用于无人机端到端避障任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch
        
        layers = [
            nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class CNNEncoder(nn.Module):
    """
    MobileNetV3 风格 CNN 编码器 (~1.8M 参数)
    
    架构:
    - Stem: Conv2d(1→32, k=3, s=2)
    - Block 1: 32 filters
    - Block 2: 64 filters, stride=2
    - Block 3: 128 filters, stride=2
    - Block 4: 256 filters, stride=2
    - 输出: 512维特征
    """
    
    def __init__(
        self,
        in_channels=1,
        stem_dim=32,
        stage_dims=(32, 64, 128, 256),
        output_dim=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU()
        )
        
        blocks = []
        prev_dim = stem_dim
        for i, dim in enumerate(stage_dims):
            stride = 2 if i > 0 else 1
            blocks.append(InvertedResidual(prev_dim, dim, stride=stride))
            prev_dim = dim
            
        self.blocks = nn.Sequential(*blocks)
        
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stage_dims[-1], output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_cnn_encoder(config):
    return CNNEncoder(
        in_channels=config.get('in_channels', 1),
        stem_dim=config.get('stem_dim', 32),
        stage_dims=config.get('stage_dims', (32, 64, 128, 256)),
        output_dim=config.get('output_dim', 512),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    encoder = CNNEncoder()
    params = encoder.get_parameter_count()
    print(f"CNN Encoder 参数量: {params:,} ({params/1e6:.2f}M)")
    
    x = torch.randn(1, 1, 60, 90)
    with torch.no_grad():
        output = encoder(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
