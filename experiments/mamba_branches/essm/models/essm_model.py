import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models')
from decision_mamba_model import CoarseSSM, FineSSM

class VisualEncoder(nn.Module):
    def __init__(self, coarse_d_state=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.proj = nn.Linear(128, 256)
        self.norm = nn.LayerNorm(256)
        self.ssm = CoarseSSM(256, coarse_d_state)

    def forward(self, x):
        x = self.stem(x).mean(dim=[2, 3])
        return self.norm(self.ssm(self.proj(x))[0] + self.proj(x))


class EssmNet(nn.Module):
    def __init__(self, coarse_d_state=32, fine_d_state=16, dropout=0.1):
        super().__init__()
        self.encoder = VisualEncoder(coarse_d_state)
        self.coarse_ssm = CoarseSSM(256, coarse_d_state)
        self.fine_ssm = FineSSM(256, fine_d_state)
        self.state_proj = nn.Linear(7, 32)
        self.fusion = nn.Sequential(
            nn.Linear(256*3+32, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 3)
        )

    def forward(self, X):
        vis = self.encoder(X[0])
        state = self.state_proj(torch.cat((X[1]*0.1, X[2]), dim=1))
        coarse, _ = self.coarse_ssm(vis)
        fine, _ = self.fine_ssm(vis)
        return self.fusion(torch.cat((vis, coarse, fine, state), dim=1)), None

def create_essm_model(config=None):
    return EssmNet(dropout=config.get('dropout', 0.1) if config else 0.1)
