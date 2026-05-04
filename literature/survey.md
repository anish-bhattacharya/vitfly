# Literature Survey: Cross-Architecture Knowledge Distillation for Robot Control

Survey date: 2026-05-04
Context: ViT+LSTM → Mamba distillation for quadrotor obstacle avoidance

---

## 1. MOHAWK (NeurIPS 2024)

**Title**: Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models  
**Authors**: Aviv Bick, Kevin Li, Eric Xing, J Zico Kolter, Albert Gu  
**Venue**: NeurIPS 2024 (Poster)  
**Links**: [arXiv:2408.10189](https://arxiv.org/abs/2408.10189), [NeurIPS Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/3848fef259495bfd04d60cdc5c1b4db7-Paper-Conference.pdf), [OpenReview](https://openreview.net/forum?id=FJlrSZBMCD)

### Core Idea
Progressive three-stage distillation from pretrained Transformer → Mamba-2. Treats both architectures as applying different forms of "mixing matrices" over token sequences, enabling alignment at multiple granularities.

### Three Stages

| Stage | What | How | Trainable |
|-------|------|-----|-----------|
| 1: Matrix Orientation | Align attention matrix ↔ SSM matrix | Minimize L2 between materialized matrices | SSM mixer params only |
| 2: Hidden-State Alignment | Align full block outputs (mixer + channel mixer) | Minimize L2 of block outputs | SSM block params |
| 3: Weight-Transfer + KD | End-to-end distillation | Transfer remaining weights (embed, norm, head) + KL/CE distillation | Full model fine-tune |

### Key Results
- **Phi-Mamba** (pure Mamba-2): 3B tokens (<1% of normal pretraining data)
- **Hybrid Phi-Mamba** (4 attention + rest Mamba-2): 5B tokens
- Ablation: Stage 1 alone gives biggest jump. Stages 2+3 refine.
- Naive output-only KD fails for cross-architecture transfer.

### Relevance to Our Work ⭐
**Most directly relevant method.** Our protocol already references MOHAWK's multi-stage design. Key takeaway: must align intermediate features (not just outputs) for cross-architecture distillation to work.

---

## 2. CAB — Cross-architecture Attention Bridge (2025)

**Title**: Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge  
**Authors**: Wang et al.  
**Links**: [arXiv:2510.19266](https://arxiv.org/abs/2510.19266), [GitHub](https://github.com/wph6/CAB)

### Core Idea
Lightweight MLP-based "bridge" that maps Transformer's explicit attention (Q, K) to Mamba's implicit attention (B, C). Enables token-level supervision across heterogeneous architectures without requiring explicit attention maps.

### Technical Details
- Mamba's SSM dynamics: `h_t = A h_{t-1} + B x_t`, `y_t = C h_t`
- B and C are input-dependent projections that implicitly encode attention-like information
- Bridge: learnable MLP projects `(B, C)` → `(Q, K)` space
- Flexible layer-wise alignment: not forced 1-to-1, can map any teacher layer to any student layer

### Key Results
- Outperforms both standard KD and other cross-architecture methods
- Data-efficient: works with limited training data
- Vision + language tasks both tested

### Relevance to Our Work ⭐
CAB's attention bridge could be adapted for our ViT+LSTM teacher. Our teacher has ViT attention layers; we can align Mamba's implicit B,C to the teacher's Q,K. This is more principled than just matching encoder features.

---

## 3. X-Distill (ICLR 2026)

**Title**: X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning  
**Authors**: Maanping Shao, Feihong Zhang, Gu Zhang, Baiye Cheng, Zhengrong Xue, Huazhe Xu  
**Venue**: ICLR 2026  
**Links**: [arXiv:2601.11269](https://arxiv.org/abs/2601.11269), [OpenReview](https://openreview.net/forum?id=9xGR0uH6NE)

### Core Idea
Offline cross-architecture distillation from DINOv2 ViT-L/14 (304M params) → ResNet-18 (11M params) on ImageNet, then joint fine-tune with diffusion policy head on robotics data.

### Two-Phase Pipeline
1. **Distillation Phase**: Freeze DINOv2 teacher, train ResNet-18 student to match [CLS] token features (MSE loss) on ImageNet-1K
2. **Fine-tune Phase**: Jointly train distilled encoder + diffusion policy head end-to-end on target task data

### Key Results
- 34 simulated benchmarks + 5 real-world manipulation tasks
- Outperforms: from-scratch ResNet, fine-tuned DINOv2, **3D point cloud encoders, VLAs**
- Key insight: distilled CNN inherits ViT's semantic feature space while keeping CNN's inductive biases

### Relevance to Our Work ⭐
**Most relevant robotics distillation paper.** Our setting is similar: offline distillation from a stronger teacher (ViT+LSTM) to efficient students (Mamba variants). X-Distill's finding that cross-architecture distillation can even surpass 3D privileged encoders is very encouraging.

---

## 4. FASD — Faster LiDAR Detection via Transformer→Mamba Distillation (2024)

**Title**: Unleashing the Potential of Mamba: Boosting a LiDAR 3D Sparse Detector by Using Cross-Model Knowledge Distillation  
**Authors**: Yurui AI et al.  
**Links**: [arXiv:2409.11018](https://arxiv.org/abs/2409.11018)

### Core Idea
Heterogeneous distillation Transformer teacher → Mamba student for LiDAR 3D object detection. Uses:
- **Dynamic Voxel Group + Adaptive Attention** for teacher (global context)
- **Adapter** for spatial feature alignment between architectures
- **Span-KD**: maps heterogeneous features into uniform logit space

### Key Results
- 4× resource reduction
- 1-2% mAP improvement over SoTA
- Waymo + nuScenes benchmarks

### Relevance
Adapter-based feature alignment is a practical technique we can adopt. Span-KD's idea of mapping features to uniform logit space is worth exploring.

---

## 5. DLRMamba — Distilling Low-Rank Mamba (2026)

**Title**: DLRMamba: Distilling Low-Rank Mamba for Edge Multispectral Fusion Object Detection  
**Links**: [arXiv:2603.06920](https://arxiv.org/abs/2603.06920)

### Core Idea
Low-rank factorization of SS2D (2D Selective Scan) for edge deployment + Structure-Aware Distillation to recover representation quality.

### Structure-Aware Distillation (Three Components)
1. **SVD-Aligned Distillation** (matrix-level): align singular components
2. **Hidden State Sequence Alignment** (dynamic-level): align state trajectories
3. **Feature Reconstruction Distillation** (output-level): align final features

### Relevance
While focused on low-rank compression (not cross-architecture), the multi-level distillation strategy (matrix → dynamics → output) echoes MOHAWK's philosophy. Useful for our future quantization / compression stages.

---

## 6. EdgeNavMamba — Mamba + KD + RL for Edge Navigation (2025)

**Title**: EdgeNavMamba: Optimized Object Detection for Edge Devices  
**Links**: [arXiv:2510.14946](https://arxiv.org/abs/2510.14946)

### Core Idea
Mamba-based object detector + knowledge distillation + RL for goal-directed navigation on edge devices (Jetson Orin Nano, Raspberry Pi 5).

### Key Results
- Student model: **67% smaller**, **73% less energy** — same accuracy as teacher
- Navigation policy: >90% success in MiniWorld
- KD loss: YOLO loss + KL(teacher_logits || student_logits, T=4) + feature matching

### Relevance
Confirms Mamba + KD works for edge navigation. Our target deployment (Intel NUC, CPU-only) has similar constraints.

---

## 7. CADiT — Channel-Aware Distillation for Nano-Drone Depth Estimation (2024)

**Title**: Distilled Depth for Nano-Drones with Channel-Aware Distillation Transformer  
**Links**: Semantic Scholar (PDF from open access)

### Core Idea
Knowledge distillation for monocular depth estimation on Crazyflie nano-drone (GAP8 processor). Proposes CADiT: Channel-Aware Distillation Transformer that lets each student channel attend to all teacher channels.

### Key Results
- Deployed on real Crazyflie nano-drone at 1.24 FPS (after quantization)
- Channel-level distillation loss outperforms standard feature matching
- Transparent objects remain a limitation

### Relevance
**Only other paper combining KD + drone obstacle avoidance.** Shows the full pipeline: train teacher → distill → quantize → deploy. Our pipeline is conceptually similar but at larger scale (Intel NUC vs GAP8).

---

## 8. KD-Mamba — Trajectory Prediction (2025)

**Title**: KD-Mamba: Selective State Space Models with Knowledge Distillation for Trajectory Prediction  
**Authors**: Shaokang Cheng, Sourav Das, Shiru Qu, Lamberto Ballan  
**Venue**: Computer Vision and Image Understanding, Vol. 261, 2025  
**Links**: [dblp](https://dblp.org/rec/journals/cviu/ChengDQB25)

### Core Idea
Mamba + knowledge distillation for trajectory prediction. Combines SSM efficiency with KD.

### Relevance
Mamba + KD applied to trajectory domain. Related methodology though different task.

---

## Synthesis: Gaps and Opportunities

| Gap | Opportunity for Our Work |
|-----|------------------------|
| No cross-architecture KD (ViT→Mamba) for robot control | **First to explore this** |
| X-Distill does ViT→CNN but not ViT→Mamba | Mamba's linear complexity is better for control |
| CADiT does nano-drone depth but not end-to-end control | We do full velocity command prediction |
| MOHAWK/CAB only tested on NLP/vision | **Robot control is novel domain** |
| Existing drone KD uses classification/depth, not BC | We do behavior cloning distillation |

Our proposed experiment fills a clear gap: **cross-architecture knowledge distillation (ViT+LSTM → Mamba) for end-to-end quadrotor obstacle avoidance**.
