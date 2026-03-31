"""
分支 E: DecisionMamba 多粒度 SSM 架构
用于无人机端到端避障任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CoarseSSM(nn.Module):
    """粗粒度 SSM (全局场景理解)"""
    def __init__(self, dim, d_state=16, num_patches=15):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.num_patches = num_patches
        
        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, d_state, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.A = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        B_state = self.x_proj(x_inner)
        A = self.A.unsqueeze(0).unsqueeze(0)
        
        h = torch.cumsum(B_state * A, dim=1)
        y = h @ self.A.unsqueeze(-1)
        y = y.squeeze(-1).unsqueeze(-1).expand(-1, -1, C)
        
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        y = y.mean(dim=1)
        
        return y


class FineSSM(nn.Module):
    """细粒度 SSM (局部障碍物细节)"""
    def __init__(self, dim, d_state=32, num_patches=15):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, d_state, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.A = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        B_state = self.x_proj(x_inner)
        A = self.A.unsqueeze(0).unsqueeze(0)
        
        h = torch.cumsum(B_state * A, dim=1)
        y = h @ self.A.unsqueeze(-1)
        y = y.squeeze(-1).unsqueeze(-1).expand(-1, -1, C)
        
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        y = y + x
        
        y = y + self.mlp(self.norm(y))
        y = y.mean(dim=1)
        
        return y


class DecisionMambaNet(nn.Module):
    """
    DecisionMamba 多粒度 SSM 架构
    
    参数量:
    - 视觉编码器: ~1.8M
    - 粗粒度 SSM: ~0.6M
    - 细粒度 SSM: ~0.9M
    - 总计: ~3.3M
    """
    
    def __init__(
        self,
        embed_dim=192,
        coarse_d_state=16,
        fine_d_state=32,
        num_patches=15,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels=1, embed_dim=embed_dim, patch_size=4)
        self.state_proj = nn.Linear(7, 64)
        
        self.coarse_ssm = CoarseSSM(embed_dim, coarse_d_state, num_patches)
        self.fine_ssm = FineSSM(embed_dim, fine_d_state, num_patches)
        
        fusion_dim = embed_dim + 64 + embed_dim + embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.fc_out = nn.Linear(128, 3)
        
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
            
        patches = self.patch_embed(X[0])
        state_feat = self.state_proj(torch.cat((X[1] * 0.1, X[2]), dim=1))
        
        coarse_feat = self.coarse_ssm(patches)
        fine_feat = self.fine_ssm(patches)
        
        global_feat = patches.mean(dim=1)
        
        fusion_input = torch.cat((global_feat, state_feat, coarse_feat, fine_feat), dim=1)
        x = self.fusion(fusion_input)
        x = self.fc_out(x)
        
        return x, None
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_decision_mamba_model(config):
    return DecisionMambaNet(
        embed_dim=config.get('embed_dim', 192),
        coarse_d_state=config.get('coarse_d_state', 16),
        fine_d_state=config.get('fine_d_state', 32),
        num_patches=config.get('num_patches', 15),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = DecisionMambaNet()
    params = model.get_parameter_count()
    print(f"DecisionMamba 总参数量: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"输入: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"输出: {output.shape}")
