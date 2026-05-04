# EdgeNavMamba: Mamba + KD + RL for Edge Navigation (2025)

**Full Title**: EdgeNavMamba: Optimized Object Detection for Edge Devices  
**Links**: [arXiv](https://arxiv.org/abs/2510.14946)

## Core Contribution
Mamba-based object detector + knowledge distillation + RL for goal-directed navigation on edge devices.

## Key Results
| Metric | Improvement |
|--------|------------|
| Model size | **67% smaller** |
| Energy/inference (Jetson Orin Nano) | **73% less** |
| Navigation success (MiniWorld) | **>90%** |
| Parameters vs baseline | **31% fewer** |

## Distillation Setup
- Teacher: Larger Mamba-based detector
- Student: Compact Mamba with LiteSS2D (weight-shared across scan directions)
- KD loss: YOLO loss + KL(T=4) + feature matching

## Implications
- ✅ Confirms Mamba + KD works for navigation on edge devices
- ✅ Deployment targets (Jetson Orin, RPi 5) similar to our Intel NUC constraint
- LiteSS2D weight-sharing idea could be explored for further compression
