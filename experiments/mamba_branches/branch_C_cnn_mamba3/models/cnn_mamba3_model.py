"""
Branch C: CNN + Mamba-3 complete model for drone obstacle avoidance

Architecture:
- CNN Encoder (~1.8M)
- Mamba-3 SSM temporal head (~1.2M)
- Total: ~3.0M

Mamba-3 core innovations:
1. Trapezoidal Discretization (second-order accurate state update)
2. Complex-valued SSM with RoPE (data-dependent rotary position encoding)
3. QK-Normalization on B, C
4. Learnable BC Bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from einops import rearrange
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


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    return x * torch.sigmoid(x)


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
    assert x.shape[1] % chunk_size == 0, f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    
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
    def __init__(self, dim, d_state=32, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = dim * expansion_factor
        self.nheads = 4
        self.headdim = self.d_inner // self.nheads
        
        assert self.d_state % 2 == 0, "d_state must be even for RoPE pairing"
        assert self.d_inner % self.nheads == 0, "d_inner must be divisible by nheads"
        
        d_in_proj = (
            2 * self.d_inner +
            2 * d_state +
            2 * self.nheads +
            d_state // 2
        )
        self.in_proj = nn.Linear(dim, d_in_proj, bias=False)
        
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.empty(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.A_log, -4, -1)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.dt_bias, 0.001, 0.1)
        
    def forward(self, x, state=None):
        batch, seqlen, _ = x.shape
        x_norm = self.norm1(x)
        
        A = -torch.exp(self.A_log)
        
        proj = self.in_proj(x_norm)
        z, x_inner, B, C, dt, lam, theta = torch.split(
            proj,
            [
                self.d_inner,
                self.d_inner,
                self.d_state,
                self.d_state,
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
            dt.unsqueeze(-1) * rearrange(theta, "b l n -> b l 1 n")
        )
        cum_angles = -torch.cumsum(raw_angles, dim=1)
        
        dA = dt * rearrange(A, "h -> 1 1 h")
        alpha = torch.exp(dA)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt
        
        x_inner = rearrange(x_inner, "b l (h p) -> b l h p", p=self.headdim)
        
        B = rearrange(B, "b l n -> b l 1 n") + self.B_bias
        C = rearrange(C, "b l n -> b l 1 n") + self.C_bias
        
        B = apply_rope(B, cum_angles)
        C = apply_rope(C, cum_angles)
        
        chunk_size = min(16, seqlen)
        if seqlen % chunk_size != 0:
            chunk_size = seqlen
        
        y_gamma, state_gamma = ssd(
            x_inner * gamma.unsqueeze(-1), dA, B, C,
            chunk_size, device=x.device,
        )
        
        B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = F.pad(x_inner[:, :-1], (0, 0, 0, 0, 1, 0))
        
        y_beta, state_beta = ssd(
            x_prev * beta.unsqueeze(-1), dA, B_prev, C,
            chunk_size, device=x.device,
        )
        
        y = y_gamma + y_beta
        ssm_state = state_gamma + state_beta
        
        y = y + x_inner * self.D.unsqueeze(-1)
        
        y = rearrange(y, "b l h p -> b l (h p)")
        y = y * silu(z)
        y = self.out_proj(y)
        
        y = y + x
        y = y + self.mlp(self.norm2(y))
        
        return y, ssm_state


class Mamba3Head(nn.Module):
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
    CNN + Mamba-3 hybrid architecture
    
    Parameters:
    - CNN Encoder: ~1.8M
    - Mamba-3 Head: ~1.2M
    - Output layer: ~1K
    - Total: ~3.0M
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
    print(f"CNN+Mamba-3 total parameters: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"Input: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"Output: {output.shape}, hidden: {hidden.shape}")
