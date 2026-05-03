# Mamba-SSM CUDA Kernel Installation
# For accelerated Mamba-2/3 training (100-300x scan speedup)

## Prerequisites
- PyTorch 2.5+ with CUDA 12.x
- CUDA toolkit (nvcc) for kernel compilation
- ~10 minutes compile time
- 8GB+ free disk space for build artifacts

## Current Status
✅ **Installed successfully (2026-05-03)**
- causal-conv1d v1.6.1 (CUDA 12.8, PyTorch 2.6 ABI)
- mamba-ssm v2.3.1 (built with --no-build-isolation)

### Install Commands (autodl environment)
```bash
source /etc/network_turbo                    # Enable network acceleration
git config --global http.sslVerify false      # Disable SSL for git
pip install mamba-ssm --no-build-isolation    # Build with system PyTorch CUDA
```

## What This Enables
- `mamba_ssm.ops.selective_scan`: CUDA-accelerated selective scan
- `mamba_ssm.modules.mamba2.Mamba2`: Production Mamba-2 block
- Mamba-3 Trapezoidal + RoPE variants

## Branches That Benefit
- **B+** (MambaVision + Mamba-3 head): Uses Mamba-3 scan heavily
- **C** (CNN + Mamba3): Uses Mamba-3 SSM
- **D** (STH-Mamba): Could switch to CUDA scan
- **E** (DecisionMamba): Could switch to CUDA scan
- **A** (VMamba SS2D): Currently pure-PyTorch sequential scan

## Current Limitation
Without compiled kernels, the SS2D in Branch A uses a slow Python
for-loop for the selective scan (~330 iterations per forward pass).
This limits throughput but is mathematically correct.

## Environment
```bash
# Current verified environment (2026-05-02)
PyTorch: 2.8.0+cu128
CUDA: 12.8
GPU: NVIDIA RTX 5090 (32GB)
Driver: 560.35.03
```
