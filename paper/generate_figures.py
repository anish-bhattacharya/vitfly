#!/usr/bin/env python3
"""Generate all thesis figures — data plots + architecture diagrams."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(f'{OUTPUT}/arch', exist_ok=True)

# ─── Style ───
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 200,
})

# ====================================================================
# FIGURE 3: Inference Latency — Bar chart + Scatter plot
# ====================================================================
def fig3_latency():
    models = ['A', 'B', 'B+', 'C', 'D', 'E', 'Teacher', 'G_basic', 'G_lstm']
    latencies = [24.3, 10.2, 9.8, 8.5, 11.5, 7.1, 9.0, 0.74, 1.00]
    params_m = [0.97, 2.61, 2.55, 2.41, 2.60, 2.19, 3.56, 0.49, 0.80]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bar chart
    colors = ['#4ECDC4' if v < 10 else '#FF6B6B' if v > 15 else '#FFE66D' for v in latencies]
    bars = ax1.bar(models, latencies, color=colors, edgecolor='gray', linewidth=0.5)
    ax1.set_ylabel('Inference Latency (ms)')
    ax1.set_xlabel('Model')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=30)
    ax1.axhline(y=16.7, color='red', linestyle='--', linewidth=0.8, label='60Hz limit (16.7ms)')
    ax1.legend()
    for bar, v in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v}', ha='center', va='bottom', fontsize=7)

    # Right: scatter
    labels = ['A', 'B', 'B+', 'C', 'D', 'E', 'Teacher', 'G_basic', 'G_lstm']
    colors_s = ['#4ECDC4' if m == 'E' else '#FF6B6B' if m == 'A' else '#45B7D1' for m in labels]
    sizes = [80 if m == 'E' else 50 for m in labels]
    for i, label in enumerate(labels):
        ax2.scatter(params_m[i], latencies[i], c=colors_s[i], s=sizes[i],
                   edgecolors='black', linewidths=0.5, zorder=5)
        offset = (0.3, 0.5) if label != 'A' else (0.3, -1.5)
        ax2.annotate(label, (params_m[i], latencies[i]),
                    (params_m[i] + offset[0], latencies[i] + offset[1]),
                    fontsize=7, ha='left')
    # Pareto frontier
    pareto_x = [0, 0.49, 2.19, 3.56]
    pareto_y = [30, 0.74, 7.1, 9.0]
    ax2.plot(pareto_x, pareto_y, '--', color='gray', linewidth=0.8, alpha=0.5)
    ax2.fill_between(pareto_x, pareto_y, 30, alpha=0.05, color='green')
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Inference Latency (ms)')
    ax2.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/figure3.pdf', bbox_inches='tight')
    plt.close()
    print('figure3.pdf done')


# ====================================================================
# FIGURE 2: Environment comparison (sphere vs trees)
# ====================================================================
def fig2_envs():
    models = ['Teacher', 'B+ Distill', 'E Distill', 'G_basic', 'G_lstm']
    sphere_5 = [2, 1, 1, 4, 4]
    sphere_7 = [5, 3, 1, 3, 6]
    trees_5 = [0, 0, 1, 0, 0]
    trees_7 = [0, 1, 2, 0, 0]

    x = np.arange(len(models))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, title, data_5, data_7 in [
        (ax1, 'Sphere Environment', sphere_5, sphere_7),
        (ax2, 'Trees Environment', trees_5, trees_7)
    ]:
        bars_5 = ax.bar(x - w/2, data_5, w, label='5 m/s', color='#45B7D1', edgecolor='gray', linewidth=0.5)
        bars_7 = ax.bar(x + w/2, data_7, w, label='7 m/s', color='#FF6B6B', edgecolor='gray', linewidth=0.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.set_ylabel('Crashes')
        ax.legend(fontsize=8)
        # Mark teacher baseline
        if 'Sphere' in title:
            ax.axhline(y=2, color='gray', linestyle=':', linewidth=0.8, label='Teacher @ 5m/s')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/figure2.pdf', bbox_inches='tight')
    plt.close()
    print('figure2.pdf done')


# ====================================================================
# FIGURE 1: Distillation framework + Main results bar chart
# ====================================================================
def fig1_overview():
    models = ['A', 'B', 'B+', 'C', 'D', 'E']
    bc = [3, 0, 3, 3, 2, 3]    # 0 = DNF for B
    distill = [3, 2, 1, 3, 2, 1]
    x = np.arange(len(models))
    w = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   gridspec_kw={'width_ratios': [1.2, 1]})

    # Left: Framework schematic
    ax1.axis('off')
    # Teacher box
    ax1.add_patch(mpatches.FancyBboxPatch((0.1, 0.65), 0.3, 0.2, boxstyle="round,pad=0.05",
                                          facecolor='#FFE66D', edgecolor='black'))
    ax1.text(0.25, 0.75, 'ViT+LSTM\nTeacher', ha='center', va='center', fontsize=9, fontweight='bold')

    # Students
    students = ['VMamba\n+LSTM', 'MambaVision\n+SSM', 'MambaVision\n+Mamba-3',
                'CNN\n+Mamba-3', 'STH\n-Mamba', 'Decision\nMamba']
    y_pos = [0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    colors_s = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF6B6B']
    for i, (s, y, c) in enumerate(zip(students, y_pos, colors_s)):
        ax1.add_patch(mpatches.FancyBboxPatch((0.1, y), 0.3, 0.08, boxstyle="round,pad=0.02",
                                              facecolor=c, edgecolor='black', alpha=0.8))
        ax1.text(0.25, y + 0.04, s, ha='center', va='center', fontsize=6)

    # Loss labels
    losses = [r'$\mathcal{L}_{feat}$', r'$\mathcal{L}_{distill}$', r'$\mathcal{L}_{GT}$']
    for i, loss in enumerate(losses):
        ax1.text(0.55, 0.72 - i * 0.08, loss, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    # Arrows (simple)
    ax1.annotate('', xy=(0.4, 0.75), xytext=(0.55, 0.72),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax1.annotate('', xy=(0.4, 0.35), xytext=(0.55, 0.64),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0, 0.9)
    ax1.set_title('(A) Distillation Framework', fontsize=10, fontweight='bold')

    # Right: Bar chart
    bars_bc = ax2.bar(x - w/2, bc, w, label='BC', color='#FF6B6B', edgecolor='gray', linewidth=0.5)
    bars_dist = ax2.bar(x + w/2, distill, w, label='Distill', color='#45B7D1', edgecolor='gray', linewidth=0.5)
    ax2.axhline(y=2, color='gray', linestyle=':', linewidth=0.8, label='Teacher')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Crashes (60m @ 5m/s)')
    ax2.set_title('(B) Main Results', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    # Add "DNF" for B BC
    ax2.text(1 - w/2, 0.2, 'DNF', ha='center', va='bottom', fontsize=7, color='red', fontweight='bold')
    # Add value labels
    for bar, v in zip(bars_bc, bc):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v}', ha='center', va='bottom', fontsize=7)
    for bar, v in zip(bars_dist, distill):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/figure1.pdf', bbox_inches='tight')
    plt.close()
    print('figure1.pdf done')


# ====================================================================
# Architecture diagrams — enhanced with layer stacking
# ====================================================================

def draw_layer_box(ax, x, y, w, h, label, color, fontsize=5.5):
    """Draw a single layer block."""
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                          facecolor=color, edgecolor='black', linewidth=0.5))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold')

def draw_stack(ax, x, y, w, h, layers, colors, total_height, label=''):
    """Draw stacked layers."""
    n = len(layers)
    layer_h = total_height / n
    for i, (layer, color) in enumerate(zip(layers, colors)):
        ly = y + total_height - (i + 1) * layer_h
        # Draw each layer with slight offset for 3D effect
        offsets = [0, -0.003, 0.003]
        for j, off in enumerate(offsets):
            alpha = 0.3 if j > 0 else 1.0
            ax.add_patch(mpatches.FancyBboxPatch((x + off, ly + off), w, layer_h - 0.005,
                                                  boxstyle="round,pad=0.01",
                                                  facecolor=color, edgecolor='black' if j == 0 else 'none',
                                                  linewidth=0.3, alpha=alpha if j > 0 else 1.0))
        # Layer label
        lines = layer.split('\n')
        label_y = ly + layer_h / 2 + (len(lines)-1) * 0.02
        for li, line in enumerate(lines):
            ax.text(x + w/2, label_y - li * 0.04, line, ha='center', va='center',
                   fontsize=4.5, fontweight='bold')

ARCH_SPECS = {
    'A': {
        'encoder_layers': ['Conv 3×3\n32ch', 'Conv 3×3\n64ch', 'Conv 3×3\n128ch', 'Conv 3×3\n256ch\n+SS2D×4'],
        'encoder_colors': ['#FFB3B3', '#FF8C8C', '#FF6B6B', '#CC4444'],
        'temporal_layers': ['LSTM\nh=128\n×3'],
        'temporal_colors': ['#FFE66D'],
        'note': '4-dir SS2D\nscan → 4608dim'
    },
    'B': {
        'encoder_layers': ['Stem\n7×7,s4', 'DWConv+\nMLP ×2', 'DWConv+\nMLP ×2', 'DWConv+\nMLP ×2'],
        'encoder_colors': ['#B3D9FF', '#7EC8E3', '#45B7D1', '#2E8BCC'],
        'temporal_layers': ['SSM\nd=16\n×2'],
        'temporal_colors': ['#96CEB4'],
        'note': 'MambaVision\nhybrid\n512dim'
    },
    'B+': {
        'encoder_layers': ['Stem\n7×7,s4', 'DWConv+\nMLP ×2', 'DWConv+\nMLP ×2', 'DWConv+\nMLP ×2'],
        'encoder_colors': ['#B3D9FF', '#7EC8E3', '#45B7D1', '#2E8BCC'],
        'temporal_layers': ['Mamba-3\nd=32', 'Mamba-3\nd=32'],
        'temporal_colors': ['#96CEB4', '#7DBFA0'],
        'note': 'MambaVision\n+ Mamba-3\n512dim'
    },
    'C': {
        'encoder_layers': ['Conv 3×3\n32ch,s2', 'Conv 3×3\n64ch,s2', 'Conv 3×3\n128ch,s2', 'Conv 3×3\n256ch\nGAP'],
        'encoder_colors': ['#B3F0D0', '#7EDCB0', '#4ECDC4', '#2EB8A0'],
        'temporal_layers': ['Mamba-3\nd=32'],
        'temporal_colors': ['#96CEB4'],
        'note': 'MobileNetV3\nstyle\n1.81M encoder'
    },
    'D': {
        'encoder_layers': ['Conv 3×3\n32ch', 'Conv 3×3\n64ch', 'Conv 3×3\n128ch', 'ST-Mamba\nscan'],
        'encoder_colors': ['#E8D5F5', '#DDB0E8', '#DDA0DD', '#C080D0'],
        'temporal_layers': ['Mamba-2\nSSD\nd=128'],
        'temporal_colors': ['#96CEB4'],
        'note': 'Spatial-temporal\nscan\n1.80M encoder'
    },
    'E': {
        'encoder_layers': ['Conv 3×3\n32ch,s2', 'Conv 3×3\n64ch,s2', 'Conv 3×3\n128ch,s2', 'Conv 3×3\n256ch\nAP'],
        'encoder_colors': ['#FFE0B3', '#FFCC80', '#FFB84D', '#FFA500'],
        'temporal_layers': ['SSM\nd=16\n×2'],
        'temporal_colors': ['#96CEB4'],
        'note': 'Light CNN\n455K encoder\n(21%)'
    },
}

def draw_arch_v2(branch):
    spec = ARCH_SPECS[branch]
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    ax.axis('off')

    # Input
    draw_layer_box(ax, 0.02, 0.35, 0.10, 0.30, 'Depth\n60×90', '#E8E8E8')

    # Encoder stack
    n_enc = len(spec['encoder_layers'])
    enc_x, enc_w = 0.16, 0.20
    enc_total_h = 0.65
    enc_y = 0.15
    draw_stack(ax, enc_x, enc_y, enc_w, enc_total_h/n_enc,
               spec['encoder_layers'], spec['encoder_colors'], enc_total_h)

    # Feature dim note
    ax.text(enc_x + enc_w + 0.01, 0.5, spec['note'], ha='left', va='center',
           fontsize=4.5, fontstyle='italic', color='gray')

    # Concat block
    draw_layer_box(ax, 0.42, 0.38, 0.08, 0.24, 'Concat\n+vel\n+quat', '#FFE66D', 5)

    # Temporal stack
    n_temp = len(spec['temporal_layers'])
    temp_x, temp_w = 0.55, 0.18
    temp_total_h = 0.50
    temp_y = 0.25
    draw_stack(ax, temp_x, temp_y, temp_w, temp_total_h/n_temp,
               spec['temporal_layers'], spec['temporal_colors'], temp_total_h)

    # Output
    draw_layer_box(ax, 0.78, 0.38, 0.18, 0.24, 'Velocity\n(vx,vy,vz)', '#E8E8E8')

    # Arrows
    for src, dst in [(0.12, 0.16), (0.36, 0.42), (0.50, 0.55), (0.73, 0.78)]:
        ax.annotate('', xy=(dst, 0.5), xytext=(src, 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Branch {branch}', fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/arch/arch_branch_{branch}.pdf', bbox_inches='tight')
    plt.close()
    print(f'arch_branch_{branch}.pdf done')


# ====================================================================
# FIGURE 4: Distributed Training-Simulation Pipeline
# ====================================================================
def fig4_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.axis('off')

    # GitHub central box
    ax.add_patch(mpatches.FancyBboxPatch((0.38, 0.40), 0.24, 0.20, boxstyle="round,pad=0.08",
                                          facecolor='#333333', edgecolor='black'))
    ax.text(0.50, 0.50, 'GitHub\nRepository', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Training Agent (left)
    ax.add_patch(mpatches.FancyBboxPatch((0.03, 0.15), 0.28, 0.65, boxstyle="round,pad=0.08",
                                          facecolor='#FFE66D', edgecolor='black', linewidth=1.5))
    ax.text(0.17, 0.75, 'Training Agent', ha='center', va='center',
           fontsize=9, fontweight='bold')
    ax.text(0.17, 0.72, '(AutoDL Cloud GPU)', ha='center', va='center',
           fontsize=7, color='gray')

    train_steps = ['① Model Architecture\n   Design', '② BC / Distill\n   Training', '③ Weight Export\n   & Commit']
    for i, step in enumerate(train_steps):
        y = 0.58 - i * 0.14
        ax.add_patch(mpatches.FancyBboxPatch((0.07, y), 0.20, 0.10, boxstyle="round,pad=0.03",
                                              facecolor='#FFF3A0', edgecolor='black', linewidth=0.5))
        ax.text(0.17, y + 0.05, step, ha='center', va='center', fontsize=6.5)

    # Simulation Agent (right)
    ax.add_patch(mpatches.FancyBboxPatch((0.69, 0.15), 0.28, 0.65, boxstyle="round,pad=0.08",
                                          facecolor='#45B7D1', edgecolor='black', linewidth=1.5))
    ax.text(0.83, 0.75, 'Simulation Agent', ha='center', va='center',
           fontsize=9, fontweight='bold')
    ax.text(0.83, 0.72, '(WSL2 + ROS + Flightmare)', ha='center', va='center',
           fontsize=7, color='gray')

    sim_steps = ['④ git pull\n   Weights', '⑤ Flightmare\n   Simulation Test', '⑥ Results\n   Summary.yaml']
    for i, step in enumerate(sim_steps):
        y = 0.58 - i * 0.14
        ax.add_patch(mpatches.FancyBboxPatch((0.73, y), 0.20, 0.10, boxstyle="round,pad=0.03",
                                              facecolor='#7EC8E3', edgecolor='black', linewidth=0.5))
        ax.text(0.83, y + 0.05, step, ha='center', va='center', fontsize=6.5)

    # Arrows between agents and GitHub
    # Training → GitHub (push)
    ax.annotate('', xy=(0.38, 0.50), xytext=(0.31, 0.50),
               arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
    ax.text(0.345, 0.53, 'push', ha='center', fontsize=6, color='#FF6B6B', fontweight='bold')

    # GitHub → Simulation (pull)
    ax.annotate('', xy=(0.73, 0.50), xytext=(0.62, 0.50),
               arrowprops=dict(arrowstyle='->', color='#45B7D1', lw=2))
    ax.text(0.675, 0.53, 'pull', ha='center', fontsize=6, color='#45B7D1', fontweight='bold')

    # Feedback arrow (bottom)
    ax.annotate('', xy=(0.55, 0.10), xytext=(0.45, 0.10),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='dashed'))
    ax.text(0.50, 0.07, 'Results → Next Iteration', ha='center', va='center',
           fontsize=7, color='gray', fontstyle='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Distributed Training-Simulation Pipeline', fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/figure4.pdf', bbox_inches='tight')
    plt.close()
    print('figure4.pdf done')


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    fig3_latency()
    fig2_envs()
    fig1_overview()
    for branch in ['A', 'B', 'B+', 'C', 'D', 'E']:
        draw_arch_v2(branch)
    fig4_pipeline()
    print('\nAll figures generated successfully!')
