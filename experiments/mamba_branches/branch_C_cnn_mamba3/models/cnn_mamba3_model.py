"""
分支 C: CNN + Mamba-3 完整模型
用于无人机端到端避障任务

架构:
- CNN 编码器 (~1.8M)
- Mamba-3 SSM 时序头 (~1.2M)
- 总计: ~3.0M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from cnn_encoder import CNNEncoder, create_cnn_encoder


class RefineInputs(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
        return X


class Mamba3Block(nn.Module):
    """Mamba-3 SSM 块"""
    def __init__(self, dim, d_state=32, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, d_state, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)
        
        self.A = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, state=None):
        B, N = x.shape[:2]
        x_norm = self.norm1(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        B_state = self.x_proj(x_inner)
        A = self.A.unsqueeze(0).unsqueeze(0)
        
        if state is None:
            state = torch.zeros(B, self.d_state, device=x.device)
            
        h = torch.cumsum(B_state * A, dim=1)
        y = h @ self.A.unsqueeze(-1)
        y = y.squeeze(-1)
        y = y.unsqueeze(-1).expand(-1, -1, self.dim)
        
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        y = y + x
        
        y = y + self.mlp(self.norm2(y))
        return y, state


class Mamba3Head(nn.Module):
    """Mamba-3 SSM 时序头 (2层)"""
    def __init__(self, input_dim, d_state=32, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.ssm_layers = nn.ModuleList([
            Mamba3Block(hidden_dim, d_state)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, hidden=None):
        x = self.input_proj(x).unsqueeze(1)
        
        states = []
        for layer in self.ssm_layers:
            x, state = layer(x)
            states.append(state)
            
        x = self.out_norm(x.squeeze(1))
        return x, torch.cat(states, dim=-1) if states else hidden


class CNNMamba3Net(nn.Module):
    """
    CNN + Mamba-3 混合架构
    
    参数量:
    - CNN Encoder: ~1.8M
    - Mamba-3 Head: ~1.2M
    - 输出层: ~1K
    - 总计: ~3.0M
    """
    
    def __init__(
        self,
        cnn_config=None,
        ssm_d_state=32,
        ssm_hidden=256,
        ssm_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        if cnn_config is None:
            cnn_config = {
                'in_channels': 1,
                'stem_dim': 32,
                'stage_dims': (32, 64, 128, 256),
                'output_dim': 512,
                'dropout': dropout
            }
        
        self.cnn = create_cnn_encoder(cnn_config)
        self.refine = RefineInputs()
        
        cnn_output = cnn_config.get('output_dim', 512)
        ssm_input = cnn_output + 3 + 4
        
        self.ssm_head = Mamba3Head(ssm_input, ssm_d_state, ssm_hidden, ssm_layers)
        self.fc_out = spectral_norm(nn.Linear(ssm_hidden, 3))
        
    def forward(self, X):
        X = self.refine(X)
        
        vision_feat = self.cnn(X[0])
        metadata = torch.cat((X[1] * 0.1, X[2]), dim=1).float()
        x = torch.cat((vision_feat, metadata), dim=1)
        
        x, hidden = self.ssm_head(x)
        x = self.fc_out(x)
        
        return x, hidden
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_cnn_mamba3_model(config):
    return CNNMamba3Net(
        cnn_config=config.get('cnn_config'),
        ssm_d_state=config.get('ssm_d_state', 32),
        ssm_hidden=config.get('ssm_hidden', 256),
        ssm_layers=config.get('ssm_layers', 2),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = CNNMamba3Net()
    params = model.get_parameter_count()
    print(f"CNN+Mamba-3 总参数量: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"输入: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"输出: {output.shape}, hidden: {hidden.shape}")
