# Gemini 绘图提示词

视觉风格统一为 CLASSIC ACCENT BAR（经典学术风格），适合本科毕业论文。

---

## Figure 4：分布式训练-仿真流水线

```
Create a CLASSIC ACCENT BAR style technical diagram for a Chinese university thesis.

VISUAL STYLE — CLASSIC ACCENT BAR:
- Horizontal section bands stacked, pale gray (#F7F7F5) fill
- Thick colored LEFT ACCENT BAR (8px) distinguishes each section
- Content boxes: white fill, thin #DDD border, 4px rounded corners
- Sans-serif typography, bold titles, regular body
- Clean, flat, zero decoration

COLOR PALETTE:
- Blue #4A90D9 — Training Agent section
- Teal #5BA58B — Simulation Agent section
- Dark #333333 — GitHub central hub
- Coral #E76F51 — Push arrows
- Slate #7B8794 — Secondary elements

LAYOUT — Three sections arranged left-to-right:

LEFT SECTION "训练Agent (AutoDL云服务器)":
- Blue left accent bar
- Three boxes stacked vertically:
  [1] "模型架构设计" (neural network layers icon)
  [2] "BC / 蒸馏训练" (GPU chip icon)
  [3] "权重提交 git push" (git branch icon)
- Downward arrows between boxes

CENTER: "GitHub 仓库"
- Dark rounded box with repository icon
- Coral arrow labeled "push weights" from Training box [3] → GitHub

RIGHT SECTION "仿真Agent (WSL2 + ROS + Flightmare)":
- Teal left accent bar
- Three boxes stacked vertically:
  [4] "拉取权重 git pull"
  [5] "Flightmare 60m赛道测试"
  [6] "结果回传 summary.yaml"
- Teal arrow from GitHub → Simulation box [4]
- Downward arrows between boxes [4]→[5]→[6]

BOTTOM: Dashed gray feedback arrow from [6] back to [1], labeled "反馈 → 下一轮迭代"

CONSTRAINTS:
- All text in Chinese
- NO clip art, NO photo elements
- Clean academic style
- Use Helvetica/Arial style font
```

---

## Arch Branch A：VMamba+LSTM

```
Create a CLASSIC ACCENT BAR style neural network architecture diagram.

VISUAL STYLE — CLASSIC ACCENT BAR:
- Left-to-right flow with 5 processing stages
- Pale gray (#F7F7F5) section backgrounds
- White content boxes, thin #DDD border, 4px rounded corners
- Clean arrows between stages

COLOR PALETTE:
- Light gray #E8E8E8 — Input/Output
- Red #FF6B6B — SS2D encoder
- Yellow #FFE66D — LSTM temporal head
- Orange #FFD93D — Concat block

STAGES (left to right):

STAGE 1 — INPUT:
Box: "Depth Image 60×90"

STAGE 2 — ENCODER (SS2D):
Title: "SS2D Encoder"
Four stacked sub-layers:
  Conv 3×3 32ch → Conv 3×3 64ch → Conv 3×3 128ch → Conv 3×3 256ch + SS2D 4-dir scan
Output: 4608-dim feature

STAGE 3 — CONCAT:
Small box: "Concat + vel(3) + quat(4)"

STAGE 4 — TEMPORAL HEAD:
Title: "LSTM"
Stacked: LSTM ×3 layers, hidden=128
Output: velocity features

STAGE 5 — OUTPUT:
Box: "Velocity (vx, vy, vz)"

Title above: "Branch A: VMamba + LSTM (0.97M params)"
Arrows between all stages. Clean academic style.
```

---

## Arch Branch B：MambaVision+SSM

```
[Same style as above, replace encoder and temporal head]

STAGE 2 — ENCODER (MambaVision):
Title: "MambaVision Encoder"
Stacked: Stem Conv7×7 s4 → DWConv+MLP ×2 → DWConv+MLP ×2 → DWConv+MLP ×2
Output: 512-dim
Note: Deep blue color #45B7D1

STAGE 4 — TEMPORAL HEAD:
Title: "SSM"
Sub-layers: SSM d_state=16 ×2 layers
Green color #96CEB4

Title above: "Branch B: MambaVision + SSM (2.61M params)"
```

---

## Arch Branch B+：MambaVision+Mamba-3

```
[Same layout as Branch B, change temporal head]

STAGE 2 — ENCODER (MambaVision):
Same as Branch B: Stem Conv7×7 → DWConv+MLP stages → 512-dim
Blue color #45B7D1

STAGE 4 — TEMPORAL HEAD:
Title: "Mamba-3"
Features: Exponential-trapezoidal discretization, dual SSD decomposition
Green color #96CEB4

Title above: "Branch B+: MambaVision + Mamba-3 (2.55M params)"
```

---

## Arch Branch C：CNN+Mamba3

```
[Same layout, CNN encoder]

STAGE 2 — ENCODER (CNN):
Title: "CNN Encoder (MobileNetV3 style)"
Stacked: Conv 3×3 32ch s2 → Conv 3×3 64ch s2 → Conv 3×3 128ch s2 → Conv 3×3 256ch + GAP
Output: 256-dim
Green-teal color #4ECDC4

STAGE 4 — TEMPORAL HEAD:
Title: "Mamba-3"
d_state=32
Green color #96CEB4

Note: 1.81M encoder params (75% of total)

Title above: "Branch C: CNN + Mamba-3 (2.41M params)"
```

---

## Arch Branch D：STH-Mamba

```
[Same layout]

STAGE 2 — ENCODER (STH-Mamba):
Title: "STH-Mamba Encoder"
Stacked: Conv 3×3 stages → Spatio-temporal scan
Output: 256-dim
Purple color #DDA0DD

STAGE 4 — TEMPORAL HEAD:
Title: "Mamba-2 (SSD)"
Multi-head, d_state=128
Green color #96CEB4

Title above: "Branch D: STH-Mamba (2.60M params)"
```

---

## Arch Branch E：DecisionMamba

```
[Same layout]

STAGE 2 — ENCODER (Light CNN):
Title: "Lightweight CNN Encoder"
Stacked: Conv 3×3 32ch s2 → Conv 3×3 64ch s2 → Conv 3×3 128ch s2 → Conv 3×3 256ch + AP
Output: 256-dim
Note: Only 455K params (21% of total)
Orange color #FFB84D

STAGE 4 — TEMPORAL HEAD:
Title: "SSM"
d_state=16 ×2 layers
Green color #96CEB4

Title above: "Branch E: DecisionMamba (2.19M params)"
```

---

## Figure 1：蒸馏框架 + 主结果

```
Create a two-panel CLASSIC ACCENT BAR figure for a Chinese thesis.

PANEL A — DISTILLATION FRAMEWORK (left half):
Top: Teacher model box "ViT+LSTM Teacher (3.56M)"
Teacher sends knowledge to 6 student models via 3 loss functions:
- L_feat (Feature Alignment)
- L_distill (Output Distillation)
- L_GT (Ground Truth)

Students (6 boxes arranged in 2×3 grid):
  Branch A: VMamba+LSTM (0.97M)
  Branch B: MambaVision+SSM (2.61M)
  Branch B+: MambaVision+Mamba-3 (2.55M)
  Branch C: CNN+Mamba-3 (2.41M)
  Branch D: STH-Mamba (2.60M)
  Branch E: DecisionMamba (2.19M)

Arrows: Teacher → [3 losses] → 6 students

PANEL B — MAIN RESULTS BAR CHART (right half):
Grouped bar chart for 6 branches:
X-axis: A, B, B+, C, D, E
Y-axis: Crashes (60m @ 5m/s)
Two bars per branch: BC (red) and Distill (blue)
Horizontal line at y=2 labeled "Teacher baseline"
B BC shows "DNF" label
B+ Distill and E Distill at y=1 (best results)

Title above: "Figure 1: Cross-architecture Distillation Framework and Main Results"
Chinese labels. Clean academic style.
```

---

## Figure 2：环境对比

```
Create a two-panel grouped bar chart in CLASSIC ACCENT BAR style.

PANEL A — SPHERE ENVIRONMENT (left):
X-axis: Teacher, B+ Distill, E Distill, G_basic, G_lstm
Y-axis: Number of Crashes
Two bars per model: 5m/s (blue #45B7D1), 7m/s (red #FF6B6B)
Teacher baseline line at y=2

PANEL B — TREES ENVIRONMENT (right):
Same axes and models
Note: Trees are easier (fewer crashes overall)
Teacher at y=0 (perfect)

Title: "Figure 2: Speed Robustness and Environment Generalization"
Highlight E Distill's speed robustness (1 crash at both speeds)
Chinese labels below charts.
```

---

## Figure 3：推理延迟

```
Create a two-panel figure combining bar chart and scatter plot.

PANEL A — INFERENCE LATENCY BAR CHART (left):
X-axis model labels: A, B, B+, C, D, E, Teacher, G_basic, G_lstm
Y-axis: Latency (ms)
Color bars by speed: green (<10ms), yellow (10-15ms), red (>15ms)
Show value labels on top of each bar
Horizontal dashed line at 16.7ms labeled "60Hz limit"
Note: E is fastest at 7.1ms, A is slowest at 24.3ms

PANEL B — PARAMS vs LATENCY SCATTER (right):
X-axis: Parameters (Millions)
Y-axis: Latency (ms)
Each model as a labeled point
Highlight E (DecisionMamba, 2.19M, 7.1ms) in red
Draw Pareto frontier curve showing E at optimal point
Note: G_basic is extremely efficient (0.49M, 0.74ms)

Title: "Figure 3: Inference Latency and Parameter Efficiency"
Chinese labels below charts.
```

---

## 使用说明

1. 把每个 prompt 复制到你用的 AI 绘图工具
2. 如果工具不支持文字渲染（如 DALL·E 3），告诉 AI "用清晰的英文/中文标注"
3. 建议用 draw.io 或 Visio 手动微调 AI 生成的图
4. 最终导出矢量 PDF 替换 `paper/figures/` 下的文件
