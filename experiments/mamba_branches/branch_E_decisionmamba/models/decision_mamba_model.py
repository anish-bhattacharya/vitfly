"""
分支 E: DecisionMamba 多粒度 SSM 架构
用于无人机端到端避障任务

参考: DecisionMamba (NeurIPS 2024) - 使用 Mamba SSM 进行离线强化学习
本实现: 多尺度 Mamba SSM 用于视觉避障
- CoarseSSM: 全局场景理解 (大 d_state, 宽上下文)
- FineSSM: 局部障碍物细节 (小 d_state, 窄上下文)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineInputs(nn.Module):
    """输入预处理模块"""
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
        return X


class CNNEncoder(nn.Module):
    """轻量级 CNN 视觉编码器 (~0.8M 参数)"""
    def __init__(self, in_channels=1, output_dim=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.conv1(x)  # (B, 32, 30, 45)
        x = self.conv2(x)  # (B, 64, 15, 23)
        x = self.conv3(x)  # (B, 128, 8, 12)
        x = self.conv4(x)  # (B, 256, 4, 6)
        x = self.pool(x)   # (B, 256, 1, 1)
        x = x.flatten(1)   # (B, 256)
        x = self.fc(x)     # (B, output_dim)
        return x


class CoarseSSM(nn.Module):
    """
    粗粒度 SSM (全局场景理解)
    
    正确的 Mamba SSM 公式:
    1. A = -exp(A_log)  # 负指数确保稳定性
    2. dt = softplus(dt_proj(x))  # 输入依赖的时间步长
    3. dA = exp(dt * A)  # 离散化 A
    4. dB = dt * B  # 离散化 B
    5. h_t = dA * h_{t-1} + dB * x_t  # 状态更新
    6. y_t = C @ h_t + D * x_t  # 输出
    """
    def __init__(self, dim, d_state=32):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # 输入投影 (x 和 z 用于门控)
        self.in_proj = nn.Linear(dim, dim * 2)
        
        # SSM 参数投影
        self.x_proj = nn.Linear(dim, d_state * 2, bias=False)  # B 和 C
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        
        # 可学习的 A 参数 (log 空间)
        self.A_log = nn.Parameter(torch.randn(d_state))
        
        # D 参数 (跳跃连接)
        self.D = nn.Parameter(torch.ones(dim) * 0.1)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, state=None):
        """
        x: (B, dim)
        state: (B, d_state) or None
        返回: (B, dim), (B, d_state)
        """
        B, C = x.shape
        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        BC = self.x_proj(x_inner)
        B_state, C_state = BC.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x_inner))
        A = -torch.exp(self.A_log)
        
        dt_mean = dt.mean(dim=-1, keepdim=True)
        dA = torch.exp(dt_mean * A.unsqueeze(0))
        dB = dt_mean * B_state
        
        if state is None:
            state = torch.zeros(B, self.d_state, device=x.device)
        state = dA * state + dB * x_inner[:, :self.d_state]
        
        y = (C_state * state).sum(dim=-1, keepdim=True)
        y = y.expand(-1, C)
        y = y + self.D * x_inner
        
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        return y, state


class FineSSM(nn.Module):
    """
    细粒度 SSM (局部障碍物细节)
    
    与 CoarseSSM 类似, 但使用更小的 d_state 和额外的 MLP
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # 额外的 MLP 用于细节提取
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, state=None):
        """
        x: (B, dim)
        state: (B, d_state) or None
        返回: (B, dim), (B, d_state)
        """
        B, C = x.shape
        x_norm = self.norm(x)
        
        xz = self.in_proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)
        
        BC = self.x_proj(x_inner)
        B_state, C_state = BC.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x_inner))
        A = -torch.exp(self.A_log)
        
        dt_mean = dt.mean(dim=-1, keepdim=True)
        dA = torch.exp(dt_mean * A.unsqueeze(0))
        dB = dt_mean * B_state
        
        if state is None:
            state = torch.zeros(B, self.d_state, device=x.device)
        state = dA * state + dB * x_inner[:, :self.d_state]
        
        y = (C_state * state).sum(dim=-1, keepdim=True)
        y = y.expand(-1, C)
        y = y + self.D * x_inner
        
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        y = y + x
        
        y = y + self.mlp(self.norm(y))
        
        return y, state


class DecisionMambaNet(nn.Module):
    """
    DecisionMamba 多粒度 SSM 架构
    
    参数量:
    - CNN 编码器: ~0.8M
    - CoarseSSM: ~0.3M
    - FineSSM: ~0.4M
    - 融合层: ~0.5M
    - 总计: ~2.0M
    """
    
    def __init__(
        self,
        embed_dim=256,
        coarse_d_state=32,
        fine_d_state=16,
        dropout=0.1
    ):
        super().__init__()
        
        self.refine = RefineInputs()
        self.cnn_encoder = CNNEncoder(in_channels=1, output_dim=embed_dim)
        self.state_proj = nn.Linear(7, embed_dim)  # vel(3) + quat(4)
        
        self.coarse_ssm = CoarseSSM(embed_dim, coarse_d_state)
        self.fine_ssm = FineSSM(embed_dim, fine_d_state)
        
        # 融合层: vision + state + coarse + fine
        fusion_dim = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.fc_out = nn.Linear(256, 3)
        
    def forward(self, X, coarse_state=None, fine_state=None):
        X = self.refine(X)
        vision_feat = self.cnn_encoder(X[0])
        state_feat = self.state_proj(torch.cat((X[1] * 0.1, X[2]), dim=1))
        
        coarse_feat, coarse_state = self.coarse_ssm(vision_feat, coarse_state)
        fine_feat, fine_state = self.fine_ssm(vision_feat, fine_state)
        
        fusion_input = torch.cat((vision_feat, state_feat, coarse_feat, fine_feat), dim=1)
        x = self.fusion(fusion_input)
        x = self.fc_out(x)
        
        return x, (coarse_state, fine_state)
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def create_decision_mamba_model(config):
    return DecisionMambaNet(
        embed_dim=config.get('embed_dim', 256),
        coarse_d_state=config.get('coarse_d_state', 32),
        fine_d_state=config.get('fine_d_state', 16),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    model = DecisionMambaNet()
    params = model.get_parameter_count()
    print(f"DecisionMamba 总参数量: {params:,} ({params/1e6:.2f}M)")
    
    X = [
        torch.randn(1, 1, 60, 90),
        torch.randn(1, 3),
        torch.randn(1, 4)
    ]
    
    with torch.no_grad():
        output, hidden = model(X)
    
    print(f"输入: depth={X[0].shape}, vel={X[1].shape}, quat={X[2].shape}")
    print(f"输出: {output.shape}")
