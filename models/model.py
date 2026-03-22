# -*- coding: utf-8 -*-
"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the models that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *
from mamba_submodules import SimplifiedSSM, SimplifiedSSMBlock, OverlapPatchMerging

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

class ConvNet(nn.Module):
    """
    Conv + FC Network 
    Num Params: 235,269
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 3)
        self.conv2 = nn.Conv2d(4, 10, 3, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        
        self.fc0 = nn.Linear(845, 256, bias=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(- self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        metadata = torch.cat((X[1]*0.1, X[2]), dim=1).float()

        x = torch.cat((x, metadata), dim=1).float()

        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x, None #None is passed to be compatible with hidden dimensions

class LSTMNet(nn.Module):
    """
    LSTM + FC Network 
    Num Params: 2,949,937
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, stride = 3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, 3,stride =  2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)

        self.lstm = LSTM(input_size=665, hidden_size=395,
                         num_layers=2, dropout=0.15, bias=False)
        self.fc1 = spectral_norm(nn.Linear(395, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 3))

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(-self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(self.bn2(F.relu(self.conv2(x))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x,X[1]*0.1, X[2]), dim=1).float()
        if len(X)>3:
            x,h = self.lstm(x, X[3])
        else:
            x,h = self.lstm(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x, h

class LSTMNetVIT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

class ViT(nn.Module):
    """
    ViT+FC Network 
    Num Params: 3,101,199   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])        
        self.decoder = nn.Linear(4608, 512)
        self.nn_fc1 = spectral_norm(nn.Linear(517, 256))
        self.nn_fc2 = spectral_norm(nn.Linear(256, 3))
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        out = F.leaky_relu(self.nn_fc1(out))
        out = self.nn_fc2(out)

        return out, None


class DroneMamba(nn.Module):
    """
    轻量化 Mamba 混合架构 for UAV 避障
    参数量目标：~2.8M (比 ViT 少 10%, 比 ViT+LSTM 少 21%)
    
    架构设计:
    - Stage 1: CNN 特征提取 (2 层)
    - Stage 2: Mamba Encoder (2 层 Simplified-SSM Block)
    - Stage 3: 特征融合与解码
    - Stage 4: 时序建模 (Lightweight LSTM 或 Temporal-SSM)
    """
    
    def __init__(self, use_temporal_ssm=False, d_state=8, hidden_size=128):
        """
        :param use_temporal_ssm: 是否使用时序 SSM 代替 LSTM
        :param d_state: SSM 的状态维度
        :param hidden_size: LSTM/SSM 的隐藏层大小
        """
        super().__init__()
        
        # Stage 1: CNN 特征提取
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)
        
        # Stage 2: Mamba Encoder (2 层 SSM Block)
        self.mamba_block1 = SimplifiedSSMBlock(32, d_state=d_state, expansion_factor=4, drop_path=0.1)
        self.mamba_block2 = SimplifiedSSMBlock(64, d_state=d_state, expansion_factor=4, drop_path=0.1)
        
        # Patch merging layers
        self.patch_merge1 = OverlapPatchMerging(1, 32, patch_size=7, stride=4, padding=3)
        self.patch_merge2 = OverlapPatchMerging(32, 64, patch_size=3, stride=2, padding=1)
        
        # Stage 3: 特征融合与解码
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.upsample = nn.Upsample(size=(8, 12), mode='bilinear', align_corners=True)
        self.fusion_conv = nn.Conv2d(80, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = spectral_norm(nn.Linear(64, 512))
        
        # Stage 4: 时序建模
        self.use_temporal_ssm = use_temporal_ssm
        if use_temporal_ssm:
            # 使用时序 SSM
            self.temporal_ssm = SimplifiedSSM(517, d_state=4, bidirectional=False)
            self.fc_out = nn.Linear(517, 3)
        else:
            # 使用轻量 LSTM
            self.lstm = nn.LSTM(input_size=517, hidden_size=hidden_size,
                               num_layers=2, dropout=0.1)
            self.fc_out = spectral_norm(nn.Linear(hidden_size, 3))
    
    def forward(self, X):
        # 处理时序批次输入：[T, B, C, H, W] -> 重塑为 [T*B, C, H, W]
        x = X[0]
        is_sequence = (len(x.shape) == 5)
        
        if is_sequence:
            T, B_seq = x.shape[:2]
            # 重塑为 [T*B, C, H, W]
            x = x.reshape(T * B_seq, *x.shape[2:])
            desvel = X[1].reshape(T * B_seq, -1) if len(X[1].shape) > 1 else X[1]
            currquat = X[2].reshape(T * B_seq, -1) if len(X[2].shape) > 1 else X[2]
        else:
            B_seq = x.shape[0]
            desvel = X[1]
            currquat = X[2]
        
        X_processed = [x, desvel, currquat]
        X_processed = refine_inputs(X_processed)
        x = X_processed[0]  # 深度图像 (B, 1, 60, 90) or (T*B, 1, 60, 90)
        
        B = x.shape[0]
        
        # Stage 1: CNN + Patch Merging
        # Layer 1
        x, H1, W1 = self.patch_merge1(x)  # (B, N1, 32), N1 = H1 * W1 = 15 * 22 = 330
        x = self.mamba_block1(x, H1, W1)  # Mamba Block 1
        
        # Layer 2
        x = x.transpose(1, 2).reshape(B, 32, H1, W1)  # (B, 32, 15, 22)
        x, H2, W2 = self.patch_merge2(x)  # (B, N2, 64), N2 = H2 * W2 = 8 * 11 = 88
        x = self.mamba_block2(x, H2, W2)  # Mamba Block 2
        
        # Stage 3: 特征融合与解码
        x = x.transpose(1, 2).reshape(B, 64, H2, W2)  # (B, 64, 8, 11)
        
        # 多尺度特征融合（类似 ViT 的设计）
        # 这里简化处理，直接使用全局池化
        x = self.global_pool(x).flatten(1)  # (B, 64)
        x = self.decoder(x)  # (B, 512)
        
        # Stage 4: 时序建模
        x = torch.cat([x, X_processed[1]/10, X_processed[2]], dim=1)  # (B, 517)
        
        if is_sequence and self.use_temporal_ssm:
            # 恢复时序维度：[T*B, 517] -> [T, B, 517]
            x = x.reshape(T, B_seq, 517)
            # 使用时序 SSM，沿时间维度处理
            outputs = []
            for t in range(T):
                x_t = x[t:t+1]  # [1, B, 517]
                x_t = self.temporal_ssm(x_t, 1, 1).squeeze(1)  # [B, 517]
                out_t = self.fc_out(x_t)  # [B, 3]
                outputs.append(out_t)
            x = torch.stack(outputs, dim=0)  # [T, B, 3]
            h = None
        elif self.use_temporal_ssm:
            # 单帧处理
            x = x.unsqueeze(1)  # (B, 1, 517)
            x = self.temporal_ssm(x, 1, 1).squeeze(1)
            x = self.fc_out(x)
            h = None
        else:
            # 使用 LSTM
            if is_sequence:
                # [T, B, 517] -> LSTM 处理
                x, h = self.lstm(x)
                x = self.fc_out(x)  # [T, B, 3]
            else:
                if len(X) > 3:
                    x, h = self.lstm(x.unsqueeze(0), X[3])
                    x = x.squeeze(0)
                else:
                    x, h = self.lstm(x.unsqueeze(0))
                    x = x.squeeze(0)
                x = self.fc_out(x)
        
        return x, h


class UNetConvLSTMNet(nn.Module):
    """
    UNet+LSTM Network 
    Num Params: 2,955,822 
    """

    def __init__(self):
        super().__init__()

        self.unet_e11 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.unet_e12 = nn.Conv2d(4, 4, kernel_size=3, padding=1) #(N, 4, 60, 90)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=3,) #(N, 4, 30, 45)

        self.unet_e21 = nn.Conv2d(4, 8, kernel_size=3, padding=1) #(N, 8, 26, 41)
        self.unet_e22 = nn.Conv2d(8, 8, kernel_size=3, padding=1) #(N, 8, 24, 39)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2,) #(N, 8, 12, 19)

        #Input: (N, 8, 12, 19)
        self.unet_e31 = nn.Conv2d(8, 16, kernel_size=3, padding=1) #(N, 8, 10, 17)
        self.unet_e32 = nn.Conv2d(16, 16, kernel_size=3, padding=1) #(N, 16, 8, 15)

        self.unet_upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,)
        self.unet_d11 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.unet_d12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.unet_upconv2 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=3,)
        self.unet_d21 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.unet_d22 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.unet_out = nn.Conv2d(4, 1, kernel_size=1)

        self.conv_conv1 = nn.Conv2d(2, 4, 5, 3)
        self.conv_conv2 = nn.Conv2d(4, 10, 5, 2)
        self.conv_avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.conv_maxpool = nn.MaxPool2d(2, 1)
        self.conv_bn1 = nn.BatchNorm2d(4)

        self.lstm = LSTM(input_size=3065, hidden_size=200, num_layers=2, dropout=0.15, bias=False)

        self.nn_fc1 = torch.nn.utils.spectral_norm(nn.Linear(200, 64))
        self.nn_fc2 = torch.nn.utils.spectral_norm(nn.Linear(64, 32))
        self.nn_fc3 = torch.nn.utils.spectral_norm(nn.Linear(32, 3))

    def forward(self, X):

        X = refine_inputs(X)

        img, des_vel, quat = X[0], X[1], X[2]
        y_e1 = torch.relu(self.unet_e12(torch.relu(self.unet_e11(img))))
        unet_enc1 = self.unet_pool1(y_e1)
        y_e2 = torch.relu(self.unet_e22(torch.relu(self.unet_e21(unet_enc1))))
        unet_enc2 = self.unet_pool2(y_e2)
        y_e3 = torch.relu(self.unet_e32(torch.relu(self.unet_e31(unet_enc2))))

        unet_dec1 = torch.relu(self.unet_d12(torch.relu(self.unet_d11(torch.cat([self.unet_upconv1(y_e3), y_e2], dim=1)))))
        unet_dec2 = torch.relu(self.unet_d22(torch.relu(self.unet_d21(torch.cat([self.unet_upconv2(unet_dec1), y_e1], dim=1)))))

        y_unet = self.unet_out(unet_dec2)
        x_conv = torch.cat((img, y_unet), dim=1)

        y_conv = -self.conv_maxpool(-torch.relu(self.conv_bn1(self.conv_conv1(x_conv))))
        y_conv = self.conv_avgpool(torch.relu(self.conv_conv2(y_conv)))

        x_lstm = torch.cat([torch.flatten(y_conv, 1), torch.flatten(y_e3, 1), des_vel*0.1, quat], dim=1).float()

        if len(X)>3:
            y_lstm, h = self.lstm(x_lstm, X[3])
        else:
            y_lstm, h = self.lstm(x_lstm)

    
        y_fc1 = F.leaky_relu(self.nn_fc1(y_lstm))
        y_fc2 = F.leaky_relu(self.nn_fc2(y_fc1))
        y = self.nn_fc3(y_fc2)

        return y, h

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    model = ConvNet().float()
    print("ConvNet: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = LSTMNet().float()
    print("LSTMNet: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = UNetConvLSTMNet().float()
    print("UNET: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = ViT().float()
    print("VIT: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = LSTMNetVIT().float()
    print("VITLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model = DroneMamba().float()
    print("DroneMamba (LSTM): ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model = DroneMamba(use_temporal_ssm=True).float()
    print("DroneMamba (SSM): ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
