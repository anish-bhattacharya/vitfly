#!/usr/bin/env python3
"""Generate all thesis figures — grayscale, school-template compliant."""

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, os

OUTPUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(f'{OUTPUT}/arch', exist_ok=True)

# ─── Grayscale template style ───
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun'],
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.2, 'grid.linestyle': '-',
})
GRAYS = ['0.15', '0.40', '0.55', '0.70', '0.85', '0.92']
HATCHES = ['///', '\\\\', 'xx', '..', '++', '||']
FIG_W = (6.75, 3.2)

# ====================================================================
# FIGURE 3: Inference Latency
# ====================================================================
def fig3_latency():
    models = ['A','B','B+','C','D','E','Teacher','G_basic','G_lstm']
    lat = [24.3,10.2,9.8,8.5,11.5,7.1,9.0,0.74,1.00]
    par = [0.97,2.61,2.55,2.41,2.60,2.19,3.56,0.49,0.80]

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=FIG_W)

    # Bar chart — grayscale+hatch
    x = np.arange(len(models))
    bars = ax1.bar(x, lat, color='white', edgecolor='black', linewidth=0.8)
    for i,b in enumerate(bars):
        b.set_hatch(HATCHES[i%len(HATCHES)])
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel('Inference Latency (ms)')
    ax1.axhline(y=16.7, color='black', linestyle='--', linewidth=0.6, label='60Hz limit')
    ax1.legend(fontsize=7)

    # Scatter — marker shapes, no color
    markers = ['o','s','^','D','v','p','*','h','<']
    for i,m in enumerate(models):
        sz = 60 if m=='E' else 35
        ax2.scatter(par[i], lat[i], s=sz, c='white', edgecolors='black',
                   linewidths=0.8, marker=markers[i], zorder=5)
        ax2.annotate(m, (par[i],lat[i]), (par[i]+0.08,lat[i]+0.5), fontsize=7)
    ax2.set_xlabel('Parameters (M)'); ax2.set_ylabel('Inference Latency (ms)')
    ax2.set_xlim(0,4.2)

    plt.tight_layout(); plt.savefig(f'{OUTPUT}/figure3.pdf'); plt.close()
    print('figure3.pdf done')

# ====================================================================
# FIGURE 2: Environment comparison
# ====================================================================
def fig2_envs():
    mods = ['Teacher','B+ Distill','E Distill','G_basic','G_lstm']
    s5 = [2,1,1,2.8,4]; s7 = [5,3,1,6,6]
    t5 = [0,0,1,2,4]; t7 = [0,1,2,3,6]

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=FIG_W,sharey=True)
    x = np.arange(len(mods)); w = 0.3

    for ax,tit,d5,d7 in [(ax1,'Sphere',s5,s7),(ax2,'Trees',t5,t7)]:
        b5 = ax.bar(x-w/2, d5, w, color='white', edgecolor='black', linewidth=0.8, hatch='///', label='5m/s')
        b7 = ax.bar(x+w/2, d7, w, color='0.6', edgecolor='black', linewidth=0.8, hatch='\\\\', label='7m/s')
        ax.set_title(tit, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(mods, fontsize=7, rotation=15)
        ax.set_ylabel('Crashes'); ax.legend(fontsize=7)

    plt.tight_layout(); plt.savefig(f'{OUTPUT}/figure2.pdf'); plt.close()
    print('figure2.pdf done')

# ====================================================================
# FIGURE 1: Distillation framework + main results
# ====================================================================
def fig1_overview():
    mods = ['A','B','B+','C','D','E']
    bc = [3,0,3,3,2,3]; dis = [3,2,1,3,2,1]
    x = np.arange(len(mods)); w = 0.3

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=FIG_W,
                                 gridspec_kw={'width_ratios':[1,1.3]})
    ax1.axis('off')
    # Framework schematic — grayscale boxes
    y0 = 0.7; bw, bh = 0.35, 0.18
    ax1.add_patch(mpatches.FancyBboxPatch((0.1,y0),bw,bh,boxstyle='round',fc='0.85',ec='black'))
    ax1.text(0.275, y0+bh/2, 'ViT+LSTM\nTeacher', ha='center', va='center', fontsize=8)
    # Loss labels
    for i,ls in enumerate([r'$L_{feat}$',r'$L_{distill}$',r'$L_{GT}$']):
        ax1.text(0.55,0.68-i*0.06,ls,fontsize=7,bbox=dict(boxstyle='round',fc='white',ec='gray'))
    # Student boxes
    sts = ['VMamba\n+LSTM','MambaV\n+SSM','MambaV\n+Mamba3','CNN+\nMamba3','STH-\nMamba','Decision\nMamba']
    for i,s in enumerate(sts):
        y = 0.55-i*0.075
        ax1.add_patch(mpatches.FancyBboxPatch((0.1,y),bw,0.06,boxstyle='round',fc=f'{0.3+i*0.08}',ec='black'))
        ax1.text(0.275,y+0.03,s,ha='center',va='center',fontsize=5.5,color='white')
    ax1.set_xlim(0,0.75); ax1.set_ylim(0,0.9)
    ax1.set_title('(A) Distillation Framework', fontsize=9)

    # Bar chart — grayscale
    b1 = ax2.bar(x-w/2, bc, w, color='white', edgecolor='black', linewidth=0.8, hatch='///', label='BC')
    b2 = ax2.bar(x+w/2, dis, w, color='0.5', edgecolor='black', linewidth=0.8, hatch='\\\\', label='Distill')
    ax2.axhline(y=2, color='black', linestyle=':', linewidth=0.7, label='Teacher')
    ax2.set_xticks(x); ax2.set_xticklabels(mods)
    ax2.set_ylabel('Crashes (60m @5m/s)'); ax2.set_title('(B) Main Results', fontsize=9)
    ax2.legend(fontsize=7)
    ax2.text(1-w/2,0.2,'DNF',ha='center',fontsize=6,color='red')

    plt.tight_layout(); plt.savefig(f'{OUTPUT}/figure1.pdf'); plt.close()
    print('figure1.pdf done')

# ====================================================================
# Architecture diagrams — grayscale drawio-compatible style
# ====================================================================
ARCHS = {
 'A':[['Conv3×3\n32ch','Conv3×3\n64ch','Conv3×3\n128ch','Conv3×3\n256ch\n+SS2D×4'],'0.7','LSTM\nh=128×3','0.85'],
 'B':[['Stem7×7\ns4','DWConv+\nMLP×2','DWConv+\nMLP×2','DWConv+\nMLP×2'],'0.75','SSM\nd=16×2','0.85'],
 'B+':[['Stem7×7\ns4','DWConv+\nMLP×2','DWConv+\nMLP×2','DWConv+\nMLP×2'],'0.75','Mamba-3\nd=32','0.85'],
 'C':[['Conv3×3\n32ch,s2','Conv3×3\n64ch,s2','Conv3×3\n128ch,s2','Conv3×3\n256ch\nGAP'],'0.8','Mamba-3\nd=32','0.85'],
 'D':[['Conv3×3\n32ch','Conv3×3\n64ch','Conv3×3\n128ch','ST-Mamba\nscan'],'0.75','Mamba-2\nSSD d=128','0.85'],
 'E':[['Conv3×3\n32ch,s2','Conv3×3\n64ch,s2','Conv3×3\n128ch,s2','Conv3×3\n256ch\nAP'],'0.7','SSM\nd=16×2','0.85'],
}

def draw_arch(branch):
    layers, ec, tname, tc = ARCHS[branch]
    fig,ax = plt.subplots(1,1,figsize=(5,2.8)); ax.axis('off')

    # Input
    ax.add_patch(mpatches.FancyBboxPatch((0.02,0.3),0.1,0.4,boxstyle='round',fc='0.9',ec='black'))
    ax.text(0.07,0.5,'Depth\n60×90',ha='center',va='center',fontsize=5.5)

    # Encoder stack
    x0, y0, bw, bh = 0.16, 0.12, 0.2, 0.14
    for i,ly in enumerate(layers):
        y = y0 + (3-i)*bh
        fc = f'{0.3+i*0.12}'
        ax.add_patch(mpatches.FancyBboxPatch((x0,y),bw,bh-0.01,boxstyle='round',fc=fc,ec='black'))
        lines = ly.split('\n')
        for li,l in enumerate(lines):
            sz = 5 if len(lines)>1 else 5.5
            ax.text(x0+bw/2, y+bh/2-0.02+(len(lines)-1)*0.025-li*0.05, l,
                   ha='center', va='center', fontsize=sz, color='white')

    # Concat
    ax.add_patch(mpatches.FancyBboxPatch((0.42,0.33),0.08,0.34,boxstyle='round',fc='0.88',ec='black'))
    ax.text(0.46,0.5,'Cat\n+vel\n+quat',ha='center',va='center',fontsize=4.5)

    # Temporal
    x1 = 0.55
    for i in range(2):
        ax.add_patch(mpatches.FancyBboxPatch((x1,0.25+i*0.2),0.18,0.18,boxstyle='round',fc='0.85',ec='black'))
    ax.text(x1+0.09,0.5,tname,ha='center',va='center',fontsize=5.5,fontweight='bold')

    # Output
    ax.add_patch(mpatches.FancyBboxPatch((0.78,0.33),0.18,0.34,boxstyle='round',fc='0.9',ec='black'))
    ax.text(0.87,0.5,'Velocity\n(vx,vy,vz)',ha='center',va='center',fontsize=5)

    # Arrows
    for a,b in [(0.12,0.16),(0.36,0.42),(0.50,0.55),(0.73,0.78)]:
        ax.annotate('',xy=(b,0.5),xytext=(a,0.5),arrowprops=dict(arrowstyle='->',color='black',lw=0.8))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f'Branch {branch}',fontsize=10,fontweight='bold')

    plt.tight_layout(); plt.savefig(f'{OUTPUT}/arch/arch_branch_{branch}.pdf'); plt.close()
    print(f'arch_branch_{branch}.pdf done')

# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    fig3_latency()
    fig2_envs()
    fig1_overview()
    for b in ['A','B','B+','C','D','E']: draw_arch(b)
    print('\nAll figures generated in grayscale!')
