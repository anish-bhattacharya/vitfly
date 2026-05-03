"""
分支 D: STH-Mamba 时空解耦架构
用于无人机端到端避障任务

架构:
- CNN 空间编码器 (~1.5M)
- Mamba-2 SSM 时序头 (~1.3M)
- 总计: ~2.8M
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


class Mamba2SSMBlock(nn.Module):
    """Mamba-2 SSM Block with proper state-space modeling"""
    def __init__(self, input_dim, d_state=16, d_inner=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_state = d_state
        self.d_inner = d_inner or input_dim * 2
        
        self.in_proj = nn.Linear(input_dim, self.d_inner * 2)
        
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)
        
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(input_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.A_log, -4, -1)
        nn.init.ones_(self.D)
        
    def forward(self, x, state=None):
        batch_size = x.shape[0]
        
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=x.device)
        
        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x_inner))
        B = self.B_proj(x_inner)
        C = self.C_proj(x_inner)
        
        A = -torch.exp(self.A_log)
        
        dA = torch.exp(dt.mean(dim=-1, keepdim=True) * A.unsqueeze(0))
        dB = dt.mean(dim=-1, keepdim=True) * B
        
        state = dA * state + dB * x_inner.mean(dim=-1, keepdim=True).expand(-1, self.d_state)
        
        y = (C * state).sum(dim=-1, keepdim=True).expand(-1, self.d_inner)
        
        y = y + self.D * x_inner
        
        y = y * torch.sigmoid(z)
        
        y = self.out_proj(y)
        y = self.dropout(y)
        
        y = y + x
        
        return y, state


class Mamba2TemporalHead(nn.Module):
    """Mamba-2 时序建模头"""
    def __init__(self, input_dim, d_state=16, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_state = d_state
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.ssm_blocks = nn.ModuleList([
            Mamba2SSMBlock(hidden_dim, d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        
        if hidden is None:
            hidden = [None] * self.num_layers
        
        new_hidden = []
        for i, block in enumerate(self.ssm_blocks):
            x, state = block(x, hidden[i])
            new_hidden.append(state)
        
        x = self.out_norm(x)
        
        return x, new_hidden


class STHMambaNet(nn.Module):
    """
    STH-Mamba 时空解耦架构
    
    参数量:
    - CNN 空间编码器: ~1.5M
    - Mamba-2 时序头: ~1.3M
    - 总计: ~2.8M
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
        self.temporal_head = Mamba2TemporalHead(
            fusion_input, temporal_d_state, temporal_hidden, temporal_layers, dropout
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
        x, hidden = self.temporal_head(fusion_input)
        x = self.fc_out(x)
        
        return x, hidden
    
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
    print(f"输出: {output.shape}, state: {len(state)} layers")
