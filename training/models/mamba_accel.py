"""
Mamba-SSM Accelerated Selective Scan
Optional CUDA-accelerated SSM using mamba_ssm library.
Falls back to pure PyTorch if mamba_ssm not installed.

Install: pip install mamba-ssm causal-conv1d --no-build-isolation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
    from mamba_ssm.modules.mamba2 import Mamba2 as Mamba2SSM
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class SelectiveScanFn(torch.autograd.Function):
    """Pure PyTorch selective scan (fallback when mamba_ssm not available)."""
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D=None, delta_softplus=True):
        if delta_softplus:
            dt = F.softplus(dt)
        batch, seq_len, dim = x.shape
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(batch, dim, A.shape[-1], device=x.device)
        outputs = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        if D is not None:
            y = y + D.unsqueeze(0).unsqueeze(0) * x
        return y


def selective_scan_mamba_fallback(x, dt, A, B, C, D=None, delta_softplus=True):
    if HAS_MAMBA_SSM:
        from mamba_ssm.ops.selective_scan import selective_scan_fn
        if x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            try:
                return selective_scan_fn(x, dt, A, B, C, D, None, delta_softplus, False)
            except Exception:
                pass
    return SelectiveScanFn.apply(x, dt, A, B, C, D, delta_softplus)


class Mamba2Block(nn.Module):
    """
    Mamba-2 block with optional CUDA acceleration.
    Falls back to pure PyTorch if mamba_ssm not installed.
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_cuda = HAS_MAMBA_SSM and torch.cuda.is_available()
        self.mamba_fallback = _Mamba2Fallback(d_model, d_state, expand)
        if self.use_cuda:
            from mamba_ssm.modules.mamba2 import Mamba2
            self.mamba_cuda = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, headdim=headdim
            )

    def forward(self, x):
        if self.use_cuda and x.is_cuda:
            return self.mamba_cuda(x)
        return self.mamba_fallback(x)


class _Mamba2Fallback(nn.Module):
    """Pure PyTorch Mamba-2 fallback (no CUDA kernels)."""
    def __init__(self, d_model, d_state=64, expand=2):
        super().__init__()
        dim = d_model * expand
        self.d_model = d_model
        self.dim = dim
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, dim * 2)
        self.dt_proj = nn.Linear(dim, dim)
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(dim, d_state, bias=False)
        self.out_proj = nn.Linear(dim, d_model)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().view(1, 1, -1)))
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_inner))
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.dim, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            B_t = self.B_proj(x_inner[:, t])
            C_t = self.C_proj(x_inner[:, t])
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dB = dt[:, t].unsqueeze(-1) * B_t.unsqueeze(-1)
            h = dA * h + dB * x_inner[:, t].unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * torch.sigmoid(z)
        return self.out_proj(y)
