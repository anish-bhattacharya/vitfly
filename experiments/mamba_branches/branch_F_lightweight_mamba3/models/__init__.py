"""
Branch F: Lightweight CNN + Mamba-3 SSM

A lightweight architecture combining a efficient CNN encoder (~0.5M params)
with the proven Mamba-3 SSM temporal head (~0.6M) for drone obstacle avoidance.
"""

from .lightweight_cnn_encoder import LightweightCNNEncoder, create_lightweight_cnn_encoder
from .mamba3_head import Mamba3Head, create_mamba3_head
from .branch_f_model import BranchFModel, create_branch_f_model
from .branch_f_v5_model import BranchFV5Model, create_branch_f_v5_model

__all__ = [
    'LightweightCNNEncoder',
    'create_lightweight_cnn_encoder',
    'Mamba3Head',
    'create_mamba3_head',
    'BranchFModel',
    'create_branch_f_model',
    'BranchFV5Model',
    'create_branch_f_v5_model',
]
