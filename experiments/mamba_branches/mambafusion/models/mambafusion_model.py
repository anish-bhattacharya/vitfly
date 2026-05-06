import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_Bplus_mambavision_mamba3/models')
sys.path.insert(0, '/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models')
from mambavision_encoder import MambaVisionEncoder, create_mambavision_encoder
from decision_mamba_model import CoarseSSM

class MambaFusion(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.vision_encoder = create_mambavision_encoder(cfg.get('mambavision_config', {}))
        self.ssm = CoarseSSM(dim=512, d_state=64)
        self.ssm_norm = nn.LayerNorm(512)
        self.fusion = nn.Sequential(
            nn.Linear(519, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3)
        )

    def forward(self, X):
        vis_feat = self.vision_encoder(X[0])
        ssm_out, _ = self.ssm(self.ssm_norm(vis_feat))
        state_in = torch.cat([ssm_out, X[1] * 0.1, X[2]], dim=1).float()
        return self.fusion(state_in), None

def create_mambafusion_model(config=None):
    return MambaFusion(config)
