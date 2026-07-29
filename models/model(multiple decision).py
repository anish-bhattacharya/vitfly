import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *

N_CANDIDATES = 10  

def refine_inputs(X):

    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        # X[2] = torch.Tensor([1, 0, 0, 0]).float()
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *

N_CANDIDATES = 8   # ← define once at the top, easy to change

def refine_inputs(X):
    if X[2] is None:
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
    return X


class LSTMNetVIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, N_CANDIDATES * 3))  # 30 values
        self.up_sample = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)

    def forward(self, X):
        X = refine_inputs(X)
        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1)
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
        out = self.nn_fc2(out)
        out = out.view(-1, N_CANDIDATES, 3) 
        return out[0], h


if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    model = LSTMNetVIT().float()
    print("VITLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
