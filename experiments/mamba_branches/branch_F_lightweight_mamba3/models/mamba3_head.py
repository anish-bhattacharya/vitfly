"""
Branch B+: Mamba-3 SSM Head (adapted from mamba3-minimal reference)
Pure PyTorch implementation of Mamba-3 SSM for temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    return x * F.sigmoid(x)


def apply_rope(x, angles):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    x_rot_even = cos_a * x1 - sin_a * x2
    x_rot_odd = sin_a * x1 + cos_a * x2
    
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


def segsum(x, device=None):
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, device=None):
    assert x.shape[1] % chunk_size == 0
    
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]
    
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    return Y, final_state


class Mamba3Block(nn.Module):
    def __init__(self, d_model, d_state=64, headdim=32, chunk_size=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        
        assert d_model % headdim == 0
        self.nheads = d_model // headdim
        assert d_state % 2 == 0
        
        self.d_inner = d_model
        self.bc_dim = d_state
        
        d_in_proj = (
            2 * self.d_inner +
            2 * self.bc_dim +
            2 * self.nheads +
            d_state // 2
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        
        self.B_norm = RMSNorm(self.bc_dim)
        self.C_norm = RMSNorm(self.bc_dim)
        
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.A_log, -4, -1)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.dt_bias, 0.001, 0.1)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        
    def forward(self, u):
        batch, seqlen, _ = u.shape
        
        A = -torch.exp(self.A_log)
        
        proj = self.in_proj(u)
        z, x, B, C, dt, lam, theta = torch.split(
            proj,
            [
                self.d_inner,
                self.d_inner,
                self.bc_dim,
                self.bc_dim,
                self.nheads,
                self.nheads,
                self.d_state // 2,
            ],
            dim=-1,
        )
        
        dt = F.softplus(dt + self.dt_bias)
        lam = torch.sigmoid(lam)
        
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        raw_angles = (
            dt.unsqueeze(-1) *
            rearrange(theta, "b l n -> b l 1 n")
        )
        cum_angles = -torch.cumsum(raw_angles, dim=1)
        
        dA = dt * rearrange(A, "h -> 1 1 h")
        alpha = torch.exp(dA)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt
        
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        
        B = rearrange(B, "b l n -> b l 1 n") + self.B_bias
        C = rearrange(C, "b l n -> b l 1 n") + self.C_bias
        
        B = apply_rope(B, cum_angles)
        C = apply_rope(C, cum_angles)
        
        y_gamma, state_gamma = ssd(
            x * gamma.unsqueeze(-1), dA, B, C,
            self.chunk_size, device=u.device,
        )
        
        B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        
        y_beta, state_beta = ssd(
            x_prev * beta.unsqueeze(-1), dA, B_prev, C,
            self.chunk_size, device=u.device,
        )
        
        y = y_gamma + y_beta
        
        y = y + x * self.D.unsqueeze(-1)
        
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y * silu(z)
        y = self.out_proj(y)
        
        return y


class Mamba3Head(nn.Module):
    def __init__(
        self,
        input_dim,
        d_state=64,
        hidden_dim=256,
        num_layers=2,
        headdim=32,
        chunk_size=32,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = RMSNorm(hidden_dim)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba3Block(hidden_dim, d_state, headdim, chunk_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.block_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.out_norm = RMSNorm(hidden_dim)
        
    def forward(self, x, hidden=None):
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch, seqlen, dim = x.shape
        
        if seqlen % self.chunk_size != 0:
            pad_len = self.chunk_size - (seqlen % self.chunk_size)
            x = F.pad(x, (0, 0, 0, pad_len))
        
        for i, (block, norm) in enumerate(zip(self.mamba_blocks, self.block_norms)):
            x = x + block(norm(x))
        
        if seqlen % self.chunk_size != 0:
            x = x[:, :seqlen, :]
        
        x = self.out_norm(x)
        
        if x.shape[1] == 1:
            x = x.squeeze(1)
        
        return x, hidden


def create_mamba3_head(config):
    return Mamba3Head(
        input_dim=config.get('input_dim', 519),
        d_state=config.get('d_state', 64),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        headdim=config.get('headdim', 32),
        chunk_size=config.get('chunk_size', 32),
        dropout=config.get('dropout', 0.1)
    )
