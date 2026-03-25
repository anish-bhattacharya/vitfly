#!/usr/bin/env python3
"""VMamba + SSM (替代 LSTM) 模型"""
import torch
import torch.nn as nn
from vmamba_encoder import VMambaEncoder, create_vmamba_encoder

class SimpleSSM(nn.Module):
    """简化 SSM 替代 LSTM"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden=None):
        x = self.input_proj(x)  # (B, hidden_dim)
        for layer in self.layers:
            x = x + layer(x)
        return x.unsqueeze(0), hidden

class VMambaSSMNet(nn.Module):
    """VMamba + SSM 架构"""
    def __init__(self, vmamba_config=None, ssm_hidden=128, ssm_layers=2, dropout=0.1):
        super().__init__()
        if vmamba_config is None:
            vmamba_config = {'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512}
        
        self.vmamba = create_vmamba_encoder(vmamba_config)
        self.ssm = SimpleSSM(input_dim=512+3+4, hidden_dim=ssm_hidden, num_layers=ssm_layers)
        self.fc_out = nn.Linear(ssm_hidden, 3)
        
    def forward(self, X, hidden_state=None):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        
        visual_feat = self.vmamba(X[0])
        fused = torch.cat([visual_feat, X[1] * 0.1, X[2]], dim=-1)
        output, hidden = self.ssm(fused, hidden_state)
        output = output.squeeze(0)
        output = self.fc_out(output)
        return output, hidden
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
    def get_vmamba_params(self):
        return sum(p.numel() for p in self.vmamba.parameters())
    def get_ssm_params(self):
        return sum(p.numel() for p in self.ssm.parameters())

if __name__ == '__main__':
    model = VMambaSSMNet()
    print(f"总参数量：{model.get_parameter_count():,}")
    print(f"VMamba: {model.get_vmamba_params():,}")
    print(f"SSM: {model.get_ssm_params():,}")
    x, v, q = torch.randn(2, 1, 60, 90), torch.randn(2, 3), torch.randn(2, 4)
    with torch.no_grad():
        out, _ = model([x, v, q])
    print(f"输出形状：{out.shape}")
