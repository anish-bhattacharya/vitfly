import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from mambavision_encoder import MambaVisionEncoder, create_mambavision_encoder
from mamba3_head import Mamba3Head, create_mamba3_head


class RefineInputs(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
        return X


class BPlusModel(nn.Module):
    def __init__(
        self,
        mambavision_config=None,
        mamba3_d_state=64,
        mamba3_hidden=256,
        mamba3_layers=2,
        mamba3_headdim=32,
        mamba3_chunk_size=32,
        dropout=0.1
    ):
        super().__init__()
        
        if mambavision_config is None:
            mambavision_config = {
                'in_channels': 1,
                'stem_dim': 48,
                'stage_dims': (64, 128, 192),
                'depths': (2, 2, 2),
                'd_state': 12,
                'dropout': dropout,
                'output_dim': 512
            }
        
        self.mambavision = create_mambavision_encoder(mambavision_config)
        self.refine = RefineInputs()
        
        vision_output = mambavision_config.get('output_dim', 512)
        mamba3_input = vision_output + 3 + 4
        
        self.mamba3_head = create_mamba3_head({
            'input_dim': mamba3_input,
            'd_state': mamba3_d_state,
            'hidden_dim': mamba3_hidden,
            'num_layers': mamba3_layers,
            'headdim': mamba3_headdim,
            'chunk_size': mamba3_chunk_size,
            'dropout': dropout
        })
        
        self.fc_out = spectral_norm(nn.Linear(mamba3_hidden, 3))
        
    def forward(self, X):
        X = self.refine(X)
        
        vision_feat = self.mambavision(X[0])
        metadata = torch.cat((X[1] * 0.1, X[2]), dim=1).float()
        x = torch.cat((vision_feat, metadata), dim=1)
        
        x, hidden = self.mamba3_head(x)
        
        if x.dim() == 3:
            x = x[:, -1, :]
        
        x = self.fc_out(x)
        
        return x, hidden
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_bplus_model(config):
    return BPlusModel(
        mambavision_config=config.get('mambavision_config'),
        mamba3_d_state=config.get('mamba3_d_state', 64),
        mamba3_hidden=config.get('mamba3_hidden', 256),
        mamba3_layers=config.get('mamba3_layers', 2),
        mamba3_headdim=config.get('mamba3_headdim', 32),
        mamba3_chunk_size=config.get('mamba3_chunk_size', 32),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = BPlusModel()
    params = model.get_parameter_count()
    print(f"Branch B+ Total Parameters: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"Input: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"Output: {output.shape}")
