"""
分支 D: STH-Mamba 时空解耦架构
用于无人机端到端避障任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    """空间编码器 (CNN)"""
    def __init__(self, in_channels=1, output_dim=256, dropout=0.1):
        super().__init__()
        
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 4, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.conv_blocks(x)
        x = self.output_proj(x)
        return x


class MambaTemporalFusion(nn.Module):
    """Mamba 时序融合器"""
    def __init__(self, input_dim, d_state=16, hidden_dim=256, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.ssm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.ssm_layers.append(nn.ModuleDict({
                'in_proj': nn.Linear(hidden_dim, hidden_dim * 2),
                'x_proj': nn.Linear(hidden_dim, d_state, bias=False),
                'out_proj': nn.Linear(hidden_dim, hidden_dim),
                'A': nn.Parameter(torch.ones(d_state)),
                'D': nn.Parameter(torch.ones(hidden_dim) * 0.1),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
            }))
            
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.d_state = d_state
        self.hidden_dim = hidden_dim
        
    def forward(self, x, state=None):
        x = self.input_proj(x).unsqueeze(1)
        B = x.shape[0]
        
        if state is None:
            state = torch.zeros(B, self.d_state, device=x.device)
            
        for layer in self.ssm_layers:
            x_norm = layer['norm1'](x)
            xz = layer['in_proj'](x_norm)
            x_inner, z = xz.chunk(2, dim=-1)
            
            B_state = layer['x_proj'](x_inner)
            A = layer['A'].unsqueeze(0).unsqueeze(0)
            
            h = torch.cumsum(B_state * A, dim=1)
            y = h @ layer['A'].unsqueeze(-1)
            y = y.squeeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            
            y = y * torch.sigmoid(z)
            y = layer['out_proj'](y)
            y = y + x
            y = y + layer['mlp'](layer['norm2'](y))
            x = y
            
        x = self.out_norm(x.squeeze(1))
        return x, state


class STHMambaNet(nn.Module):
    """
    STH-Mamba 时空解耦架构
    
    参数量:
    - CNN 空间编码器: ~1.5M
    - Mamba 时序融合器: ~1.8M
    - 总计: ~3.3M
    """
    
    def __init__(
        self,
        spatial_dim=256,
        temporal_d_state=16,
        temporal_hidden=256,
        temporal_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim, dropout=dropout)
        self.state_proj = nn.Linear(7, 64)
        
        fusion_input = spatial_dim + 64
        self.temporal_fusion = MambaTemporalFusion(
            fusion_input, temporal_d_state, temporal_hidden, temporal_layers
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(temporal_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
            
        spatial_feat = self.spatial_encoder(X[0])
        state_feat = self.state_proj(torch.cat((X[1] * 0.1, X[2]), dim=1))
        
        fusion_input = torch.cat((spatial_feat, state_feat), dim=1)
        x, state = self.temporal_fusion(fusion_input)
        x = self.fc_out(x)
        
        return x, state
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_sth_mamba_model(config):
    return STHMambaNet(
        spatial_dim=config.get('spatial_dim', 256),
        temporal_d_state=config.get('temporal_d_state', 16),
        temporal_hidden=config.get('temporal_hidden', 256),
        temporal_layers=config.get('temporal_layers', 3),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = STHMambaNet()
    params = model.get_parameter_count()
    print(f"STH-Mamba 总参数量: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, state = model(X)
    
    print(f"输入: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"输出: {output.shape}, state: {state.shape}")
