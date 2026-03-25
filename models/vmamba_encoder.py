"""
分支 A: VMamba 视觉编码器 (裁剪版)
用于无人机端到端避障任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SS2D(nn.Module):
    """
    2D 选择性扫描模块 (简化高效版)
    使用向量化操作代替循环，提高并行效率
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # 投影层
        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, d_state, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        # 可学习参数
        self.A = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        N = H * W
        
        # 投影
        x_flat = x.view(B, N, C)
        xz = self.in_proj(x_flat)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 状态空间投影
        B_state = self.x_proj(x_inner)  # (B, N, d_state)
        
        # 向量化 SSM 计算 (代替循环)
        A = self.A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_state)
        
        # 累积和计算状态
        # h_t = sum(A^(t-k) * B_k * x_k for k=0..t)
        # 使用累积和近似
        h = torch.cumsum(B_state * A, dim=1)  # (B, N, d_state)
        
        # 输出投影
        y = h @ self.A.unsqueeze(-1)  # (B, N, 1) -> 简化
        y = y.squeeze(-1) * self.D[:1]  # (B, N)
        y = y.unsqueeze(-1).expand(-1, -1, C)  # (B, N, C)
        
        # 门控输出
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        
        # 残差连接
        y = y + x_flat
        
        return y.view(B, H, W, C)


class VSSBlock(nn.Module):
    """视觉状态空间块"""
    def __init__(self, dim, d_state=16, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """图像分块嵌入"""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class VMambaEncoder(nn.Module):
    """裁剪版 VMamba 视觉编码器"""
    
    def __init__(self, in_channels=1, embed_dim=64, depth=4, d_state=16, dropout=0.0, output_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            VSSBlock(embed_dim, d_state, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
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
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=[1, 2])
        x = self.output_proj(x)
        return x
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_vmamba_encoder(config):
    return VMambaEncoder(
        in_channels=config.get('in_channels', 1),
        embed_dim=config.get('embed_dim', 64),
        depth=config.get('depth', 4),
        d_state=config.get('d_state', 16),
        dropout=config.get('dropout', 0.0),
        output_dim=config.get('output_dim', 512)
    )


if __name__ == '__main__':
    encoder = VMambaEncoder(embed_dim=64, depth=4, d_state=16)
    params = encoder.get_parameter_count()
    print(f"VMamba Encoder 参数量：{params:,}")
    
    x = torch.randn(1, 1, 60, 90)
    with torch.no_grad():
        output = encoder(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    
    configs = [
        {'embed_dim': 48, 'depth': 2, 'd_state': 8},
        {'embed_dim': 64, 'depth': 4, 'd_state': 16},
        {'embed_dim': 96, 'depth': 6, 'd_state': 32},
    ]
    
    print("\n不同配置的参数量:")
    for cfg in configs:
        enc = create_vmamba_encoder(cfg)
        params = enc.get_parameter_count()
        print(f"  dim={cfg['embed_dim']}, depth={cfg['depth']}, d_state={cfg['d_state']}: {params:,}")
