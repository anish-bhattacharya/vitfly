import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SS2D(nn.Module):
    def __init__(self, dim=64, d_state=16, dt_rank="auto", d_conv=3, dropout=0.0):
        super().__init__()
        self.d_model = dim
        self.d_state = d_state
        self.d_inner = dim
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank
        self.k_group = 4
        
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.act = nn.SiLU()
        
        self.conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        self.x_proj_weight = nn.Parameter(
            torch.stack([
                nn.Linear(dim, self.dt_rank + d_state * 2, bias=False).weight
                for _ in range(self.k_group)
            ], dim=0)
        )
        
        self.dt_projs_weight = nn.Parameter(
            0.1 * torch.randn(self.k_group, dim, self.dt_rank)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.exp(
                torch.rand(self.k_group, dim) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
        )
        with torch.no_grad():
            self.dt_projs_bias.copy_(self.dt_projs_bias + torch.log(-torch.expm1(-self.dt_projs_bias)))
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(dim, 1)
        self.A_logs = nn.Parameter(torch.log(A).repeat(self.k_group, 1, 1).flatten(0, 1))
        self.A_logs._no_weight_decay = True
        
        self.Ds = nn.Parameter(torch.ones(self.k_group * dim))
        self.Ds._no_weight_decay = True
        
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def selective_scan_pytorch(self, x, dt, A, B, C, D):
        B_batch, K, D_inner, L = x.shape
        N = A.shape[-1]
        
        dt = F.softplus(dt + self.dt_projs_bias.view(1, K, D_inner, 1))
        
        dA = torch.exp(A.view(K, D_inner, N, 1) * dt.view(B_batch, K, D_inner, 1, L))
        
        h = torch.zeros(B_batch, K, D_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            dB_t = dt[:, :, :, t].unsqueeze(-1) * B[:, :, :, t].unsqueeze(2)
            h = dA[:, :, :, :, t] * h + dB_t * x[:, :, :, t].unsqueeze(-1)
            y_t = (h * C[:, :, :, t].unsqueeze(2)).sum(dim=-1)
            ys.append(y_t)
        
        y = torch.stack(ys, dim=-1)
        y = y + x * D.view(1, K, D_inner, 1)
        
        return y
    
    def forward(self, x):
        B, H, W, C = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        z = self.act(z)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        
        L = H * W
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        
        out_y = self.selective_scan_pytorch(
            xs.float(), dts.float(), As, Bs.float(), Cs.float(), Ds
        )
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class VSSBlock(nn.Module):
    def __init__(self, dim=64, d_state=16, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim=dim, d_state=d_state, dropout=dropout)
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


class VMambaEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, depth=4, d_state=16, dropout=0.1, output_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=4)
        self.blocks = nn.ModuleList([
            VSSBlock(dim=embed_dim, d_state=d_state, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim

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
        output_dim=config.get('output_dim', 512),
    )
