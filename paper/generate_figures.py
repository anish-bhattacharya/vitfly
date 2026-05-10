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
# Architecture diagrams (arch_branch_A through E)
# ====================================================================
ARCH_DATA = {
    'A': {'encoder': 'SS2D\n(Mamba Scan)', 'encoder_color': '#FF6B6B', 'temporal': 'LSTM\n(State)', 'temporal_color': '#FFE66D'},
    'B': {'encoder': 'MambaVision\n(Conv+MLP)', 'encoder_color': '#45B7D1', 'temporal': 'SSM\n(d_state=16)', 'temporal_color': '#96CEB4'},
    'B+': {'encoder': 'MambaVision\n(Conv+MLP)', 'encoder_color': '#45B7D1', 'temporal': 'Mamba-3', 'temporal_color': '#96CEB4'},
    'C': {'encoder': 'CNN\n(MobileNetV3)', 'encoder_color': '#4ECDC4', 'temporal': 'Mamba-3\n(d_state=32)', 'temporal_color': '#96CEB4'},
    'D': {'encoder': 'CNN-like\nEncoder', 'encoder_color': '#4ECDC4', 'temporal': 'Mamba-2\n(SSD)', 'temporal_color': '#96CEB4'},
    'E': {'encoder': 'Light CNN\n(455K)', 'encoder_color': '#4ECDC4', 'temporal': 'SSM\n(d_state=16)', 'temporal_color': '#96CEB4'},
}

def draw_arch(branch, data):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.axis('off')

    # Depth image input
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.35), 0.12, 0.3, boxstyle="round,pad=0.03",
                                          facecolor='#E8E8E8', edgecolor='black'))
    ax.text(0.08, 0.5, 'Depth\n60×90', ha='center', va='center', fontsize=6)

    # Arrow
    ax.annotate('', xy=(0.15, 0.5), xytext=(0.14, 0.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Encoder
    ax.add_patch(mpatches.FancyBboxPatch((0.16, 0.2), 0.2, 0.6, boxstyle="round,pad=0.05",
                                          facecolor=data['encoder_color'], edgecolor='black', alpha=0.85))
    ax.text(0.26, 0.5, data['encoder'], ha='center', va='center', fontsize=7, fontweight='bold')

    # Feature concat (small)
    ax.add_patch(mpatches.FancyBboxPatch((0.4, 0.35), 0.08, 0.3, boxstyle="round,pad=0.02",
                                          facecolor='#FFE66D', edgecolor='black'))
    ax.text(0.44, 0.5, 'Cat\n+vel\n+quat', ha='center', va='center', fontsize=5)

    # Arrow
    ax.annotate('', xy=(0.38, 0.5), xytext=(0.36, 0.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.48, 0.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Temporal head
    ax.add_patch(mpatches.FancyBboxPatch((0.52, 0.25), 0.2, 0.5, boxstyle="round,pad=0.05",
                                          facecolor=data['temporal_color'], edgecolor='black', alpha=0.85))
    ax.text(0.62, 0.5, data['temporal'], ha='center', va='center', fontsize=7, fontweight='bold')

    # Arrow
    ax.annotate('', xy=(0.74, 0.5), xytext=(0.72, 0.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Output
    ax.add_patch(mpatches.FancyBboxPatch((0.76, 0.38), 0.2, 0.24, boxstyle="round,pad=0.03",
                                          facecolor='#E8E8E8', edgecolor='black'))
    ax.text(0.86, 0.5, 'Velocity\n(vx,vy,vz)', ha='center', va='center', fontsize=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Branch {branch}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/arch/arch_branch_{branch}.pdf', bbox_inches='tight')
    plt.close()
    print(f'arch_branch_{branch}.pdf done')


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    fig3_latency()
    fig2_envs()
    fig1_overview()
    for branch, data in ARCH_DATA.items():
        draw_arch(branch, data)
    print('\nAll figures generated successfully!')
