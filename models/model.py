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
        self.lstm = (nn.LSTM(input_size=519, hidden_size=128,
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


class TeacherVITLSTM(nn.Module):
    """
    ViT+LSTM teacher — checkpoint-compatible version (lstm input_size=517).
    
    The upstream best model (7m/s real flight, ViTLSTM_model.pth) was trained with
    scalar desired_vel (1-dim) rather than full 3D velocity. Input layout:
      512 (visual feat) + 1 (desired_vel/10) + 4 (quaternion) = 517
    
    LSTMNetVIT above uses input_size=519 (512+3+4) and CANNOT load the upstream
    checkpoint. Use this class instead when loading ViTLSTM_model.pth.
    """
    def __init__(self):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, 
                                       n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, 
                                       n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))
        
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
        # Scalar desired_vel (X[1][:, :1]), not full 3D — matches upstream checkpoint
        out = torch.cat([out, X[1][:, :1] / 10.0, X[2]], dim=1).float()
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
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
            self.lstm = nn.LSTM(input_size=519, hidden_size=hidden_size,
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
        x = torch.cat([x, X_processed[1]/10, X_processed[2]], dim=1)  # (B, 519)
        
        if is_sequence and self.use_temporal_ssm:
            # 恢复时序维度：[T*B, 517] -> [T, B, 519]
            x = x.reshape(T, B_seq, 519)
            # 使用时序 SSM，沿时间维度处理
            outputs = []
            for t in range(T):
                x_t = x[t:t+1]  # [1, B, 519]
                x_t = self.temporal_ssm(x_t, 1, 1).squeeze(1)  # [B, 517]
                out_t = self.fc_out(x_t)  # [B, 3]
                outputs.append(out_t)
            x = torch.stack(outputs, dim=0)  # [T, B, 3]
            h = None
        elif self.use_temporal_ssm:
            # 单帧处理
            x = x.unsqueeze(1)  # (B, 1, 519)
            x = self.temporal_ssm(x, 1, 1).squeeze(1)
            x = self.fc_out(x)
            h = None
        else:
            # 使用 LSTM
            if is_sequence:
                # [T, B, 519] -> LSTM 处理
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


class VMambaLSTMNet(nn.Module):
    """VMamba + LSTM 架构 (分支 A 最佳模型)"""
    def __init__(self):
        super().__init__()
        # VMamba 视觉编码器
        self.vmamba_config = {'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512}
        self.vmamba = self._create_vmamba(self.vmamba_config)
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(input_size=519, hidden_size=128, num_layers=2, dropout=0.1, bias=False)
        self.fc_out = nn.Linear(128, 3)
        
    def _create_vmamba(self, config):
        # 简化的 VMamba 编码器（实际应从 vmamba_encoder.py 导入）
        from vmamba_encoder import VMambaEncoder
        return VMambaEncoder(
            in_channels=1,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            d_state=config['d_state'],
            dropout=0.1,
            output_dim=config['output_dim']
        )
    
    def forward(self, X, hidden_state=None):
        # 预处理
        X = refine_inputs(X)
        
        # 视觉编码
        visual_feat = self.vmamba(X[0])  # (B, 512)
        
        # 特征融合
        fused = torch.cat([visual_feat, X[1]*0.1, X[2]], dim=-1)  # (B, 519)
        
        # LSTM
        if hidden_state is None:
            fused_seq = fused.unsqueeze(0)
            output, (h_n, c_n) = self.lstm(fused_seq)
            output = output.squeeze(0)
            hidden_state = (h_n, c_n)
        else:
            fused_seq = fused.unsqueeze(0)
            output, hidden_state = self.lstm(fused_seq, hidden_state)
            output = output.squeeze(0)
        
        output = self.fc_out(output)
        return output, hidden_state


class CNNMamba3(nn.Module):
    """
    分支C：CNN空间编码 + Mamba-3时序架构
    目标：极致轻量化(<1M参数)下的性能底线
    
    架构特点：
    - Stage 1: MobileNetV3浅层CNN (2-6层可调)
    - Stage 2: Mamba-3 SSM × 2 (d_state=32-128可调)
    - Stage 3: 轻量特征融合与解码
    - Stage 4: 时序建模 (可选LSTM或Temporal-SSM)
    
    参数量：目标 < 1M (当前实现 ~2.14M)
    """
    
    def __init__(self, cnn_depth=4, ssm_layers=2, d_state=32, hidden_size=128, use_temporal_ssm=False):
        """
        :param cnn_depth: CNN层数 [2,4,6]
        :param ssm_layers: Mamba-3 SSM层数 [1,2,4]
        :param d_state: SSM状态维度 [32,64,128]
        :param hidden_size: LSTM隐藏层大小
        :param use_temporal_ssm: 是否使用时序SSM代替LSTM
        """
        super().__init__()
        
        # Stage 1: MobileNetV3浅层CNN
        self.cnn_depth = cnn_depth
        self.d_state = d_state
        self.cnn_layers = nn.ModuleList()
        
        # 输入: (B, 1, 60, 90)
        in_channels = 1
        out_channels = 32
        
        # CNN层配置 - 增加容量但仍保持轻量
        for i in range(cnn_depth):
            if i == 0:
                # 第一层：较大卷积核，提取基础特征
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
            elif i == 1:
                # 第二层：深度可分离卷积，效率高
                conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
                    nn.Conv2d(out_channels, out_channels*2, kernel_size=1, stride=1)
                )
                out_channels *= 2
            elif i == 2:
                # 第三层：扩张卷积，增加感受野
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
            else:
                # 后续层：轻量卷积
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
            self.cnn_layers.append(conv)
            self.cnn_layers.append(nn.BatchNorm2d(out_channels))
            self.cnn_layers.append(nn.GELU())
            
            if i % 2 == 0 and i > 0:
                # 每两层添加池化，逐步降采样
                self.cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.cnn_out_channels = out_channels
        
        # Stage 2: Mamba-3 SSM
        self.ssm_layers = ssm_layers
        self.ssm_blocks = nn.ModuleList()
        
        for i in range(ssm_layers):
            ssm_block = SimplifiedSSMBlock(
                self.cnn_out_channels, 
                d_state=d_state,
                expansion_factor=2,  # 轻量化设计
                drop_path=0.05
            )
            self.ssm_blocks.append(ssm_block)
        
        # Patch merging for SSM
        self.patch_merge = OverlapPatchMerging(
            in_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            patch_size=3,
            stride=2,
            padding=1
        )
        
        # Stage 3: 轻量特征融合与解码
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Sequential(
            nn.Linear(self.cnn_out_channels, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)  # 输出128维以匹配LSTM输入
        )
        
        # Stage 4: 时序建模
        self.use_temporal_ssm = use_temporal_ssm
        if use_temporal_ssm:
            # 使用时序SSM
            self.temporal_ssm = SimplifiedSSM(129, d_state=16, bidirectional=False)
            self.fc_out = nn.Linear(129, 3)
        else:
            # 使用轻量LSTM
            self.lstm = nn.LSTM(
                input_size=129,  # 128 + 1 (desvel/10)
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=False
            )
            self.fc_out = nn.Linear(hidden_size, 3)
    
    def forward(self, X):
        # 处理输入
        x = X[0]
        is_sequence = (len(x.shape) == 5)
        
        if is_sequence:
            T, B_seq = x.shape[:2]
            x = x.reshape(T * B_seq, *x.shape[2:])
            desvel = X[1].reshape(T * B_seq, -1) if len(X[1].shape) > 1 else X[1]
            currquat = X[2].reshape(T * B_seq, -1) if len(X[2].shape) > 1 else X[2]
        else:
            B_seq = x.shape[0]
            desvel = X[1]
            currquat = X[2]
        
        X_processed = [x, desvel, currquat]
        X_processed = refine_inputs(X_processed)
        x = X_processed[0]
        
        B = x.shape[0]
        
        # Stage 1: CNN特征提取
        for layer in self.cnn_layers:
            x = layer(x)
        
        # 获取CNN输出尺寸
        _, C_cnn, H_cnn, W_cnn = x.shape
        
        # Stage 2: Mamba-3 SSM处理
        # Patch merging - 将空间特征转换为序列
        x, H_patch, W_patch = self.patch_merge(x)  # (B, N, C_cnn)
        
        # SSM处理
        for ssm_block in self.ssm_blocks:
            x = ssm_block(x, H_patch, W_patch)
        
        # 转换回空间格式
        x = x.transpose(1, 2).reshape(B, self.cnn_out_channels, H_patch, W_patch)
        
        # Stage 3: 特征融合与解码
        x = self.global_pool(x).flatten(1)  # (B, C_cnn)
        x = self.decoder(x)  # (B, 64)
        
        # 融合速度信息
        desvel_normalized = X_processed[1] / 10.0
        x = torch.cat([x, desvel_normalized], dim=1)  # (B, 129)
        
        # Stage 4: 时序建模
        if is_sequence:
            # 恢复时序维度
            x = x.reshape(T, B_seq, 129)
            
            if self.use_temporal_ssm:
                # 时序SSM处理 - 需要添加空间维度
                # x形状: (T, B, 129) -> 需要转换为 (T, B, 1, 129) 然后展平
                x_reshaped = x.unsqueeze(2)  # (T, B, 1, 129)
                B_total = T * B_seq
                x_reshaped = x_reshaped.reshape(B_total, 1, 129)  # (T*B, 1, 129)
                x_processed = self.temporal_ssm(x_reshaped, 1, 1)  # (T*B, 1, 129)
                x_processed = x_processed.reshape(T, B_seq, 129)  # 恢复形状
                x = x_processed.mean(dim=0)  # 池化时间维度
                output = self.fc_out(x)
            else:
                # LSTM处理
                x, (h_n, c_n) = self.lstm(x)
                x = x[-1]
                output = self.fc_out(x)
        else:
            # 非时序模式
            h_n, c_n = None, None
            if self.use_temporal_ssm:
                # 添加空间维度以匹配SSM输入格式
                x = x.unsqueeze(1)  # (B, 1, 129)
                x = self.temporal_ssm(x, 1, 1)  # (B, 1, 129)
                x = x.squeeze(1)  # (B, 129)
                output = self.fc_out(x)
            else:
                # 单步LSTM
                x = x.unsqueeze(0)
                x, (h_n, c_n) = self.lstm(x)
                x = x.squeeze(0)
                output = self.fc_out(x)
        
        # 返回输出和hidden_state（为兼容性）
        hidden_state = None
        if not self.use_temporal_ssm and is_sequence:
            hidden_state = (h_n, c_n)
        
        return output, hidden_state
    
    def count_parameters(self):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"CNN深度: {self.cnn_depth}, SSM层数: {self.ssm_layers}, d_state: {self.d_state}")
        print(f"是否达到<1M目标: {'是' if total_params < 1000000 else '否'}")
        
        return total_params, trainable_params
