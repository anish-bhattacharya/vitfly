# Mamba-SSM CUDA Kernel Installation
# For accelerated Mamba-2/3 training (100-300x scan speedup)

## Prerequisites
- PyTorch 2.5+ with CUDA 12.x
- CUDA toolkit (nvcc) for kernel compilation
- ~10 minutes compile time
- 8GB+ free disk space for build artifacts

## Install
```bash
# Install causal convolution (dependency)
# NOTE: Must match PyTorch's CUDA version
# Current env: PyTorch 2.8.0+cu128, CUDA 12.8
# If pip build fails with CUDA version mismatch, try:
pip install causal-conv1d --no-build-isolation

# If still failing, install from pre-compiled wheels:
# https://github.com/Dao-AILab/causal-conv1d/releases
pip install causal-conv1d --find-links https://github.com/Dao-AILab/causal-conv1d/releases

# Install mamba-ssm (compiles CUDA kernels)
pip install mamba-ssm
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
