"""
分支 B: MambaVision + SSM 完整模型
用于无人机端到端避障任务

架构:
- MambaVision 视觉编码器 (~2.5M)
- SSM 时序头 (~0.8M)
- 总计: ~3.3M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from mambavision_encoder import MambaVisionEncoder, create_mambavision_encoder


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


class SSMHead(nn.Module):
    """SSM 时序建模头 (替代LSTM)"""
    def __init__(self, input_dim, d_state=16, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_state = d_state
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for i in range(num_layers)
        ])
        
        self.state_proj = nn.Linear(input_dim, d_state * num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, hidden=None):
        """
        x: (B, input_dim)
        hidden: (B, d_state * num_layers) or None
        """
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.d_state * self.num_layers, device=x.device)
        
        state = self.state_proj(x)
        
        for i, layer in enumerate(self.ssm_layers):
            h_i = hidden[:, i*self.d_state:(i+1)*self.d_state]
            x = layer(x) + h_i.mean(dim=-1, keepdim=True).expand(-1, self.hidden_dim)
            
        x = self.out_norm(x)
        return x, hidden


class MambaVisionSSMNet(nn.Module):
    """
    MambaVision + SSM 混合架构
    
    参数量:
    - MambaVision Encoder: ~2.5M
    - SSM Head: ~0.8M
    - 输出层: ~2K
    - 总计: ~3.3M
    """
    
    def __init__(
        self,
        mambavision_config=None,
        ssm_d_state=16,
        ssm_hidden=256,
        ssm_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        if mambavision_config is None:
            mambavision_config = {
                'in_channels': 1,
                'stem_dim': 64,
                'stage_dims': (96, 192, 384),
                'depths': (4, 4, 4),
                'd_state': 16,
                'dropout': dropout,
                'output_dim': 512
            }
        
        self.mambavision = create_mambavision_encoder(mambavision_config)
        self.refine = RefineInputs()
        
        vision_output = mambavision_config.get('output_dim', 512)
        ssm_input = vision_output + 3 + 4
        
        self.ssm_head = SSMHead(ssm_input, ssm_d_state, ssm_hidden, ssm_layers)
        self.fc_out = spectral_norm(nn.Linear(ssm_hidden, 3))
        
    def forward(self, X):
        X = self.refine(X)
        
        vision_feat = self.mambavision(X[0])
        metadata = torch.cat((X[1] * 0.1, X[2]), dim=1).float()
        x = torch.cat((vision_feat, metadata), dim=1)
        
        x, hidden = self.ssm_head(x)
        x = self.fc_out(x)
        
        return x, hidden
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_mambavision_ssm_model(config):
    return MambaVisionSSMNet(
        mambavision_config=config.get('mambavision_config'),
        ssm_d_state=config.get('ssm_d_state', 16),
        ssm_hidden=config.get('ssm_hidden', 256),
        ssm_layers=config.get('ssm_layers', 2),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = MambaVisionSSMNet()
    params = model.get_parameter_count()
    print(f"MambaVision+SSM 总参数量: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"输入: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"输出: {output.shape}, hidden: {hidden.shape}")
