"""
@authors: A Bhattacharya, et. al (modified by Lingma)
@organization: GRASP Lab, University of Pennsylvania
@date: 2026-03-16
@license: ...

@brief: Simplified State Space Model (SSM) modules for DroneMamba architecture
        Optimized for UAV obstacle avoidance tasks

@source: Inspired by Mamba/Vim architectures, simplified for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedSSM(nn.Module):
    """
    简化的状态空间模型，专为 UAV 避障优化
    
    核心思想:
    - 使用标量对角矩阵 A 代替全连接矩阵（减少参数）
    - 共享时间步长的 B, C 投影（降低内存占用）
    - 双向扫描机制捕捉长程依赖
    - 引入门控机制增强表达能力
    
    输入格式：(B, N, C) where N = H * W
    """
    
    def __init__(self, channels, d_state=8, bidirectional=True):
        super().__init__()
        self.d_state = d_state
        self.bidirectional = bidirectional
        
        # 对角 SSM 参数（标量，极大减少参数）
        # A 初始化为 1，表示单位动态
        self.A = nn.Parameter(torch.ones(channels, d_state))
        
        # B 和 C 的投影层（将输入映射到状态空间）
        self.B_proj = nn.Linear(channels, d_state, bias=False)
        self.C_proj = nn.Linear(channels, d_state, bias=False)
        
        # Delta 参数（用于离散化）
        self.delta = nn.Parameter(torch.ones(channels, d_state))
        
        # 输入门控（增强非线性，增强对障碍物边缘的敏感性）
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        
        # 输出投影
        out_channels = d_state * (2 if bidirectional else 1)
        self.out_proj = nn.Linear(out_channels, channels)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以提高训练稳定性"""
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, H, W):
        """
        :param x: tensor with shape (B, N, C) where
            B is the batch size
            N is the number of tokens (H * W)
            C is the number of channels
        :param H, W: spatial dimensions
        :return: tensor with shape (B, N, C)
        """
        B, N, C = x.shape
        
        # 应用门控机制
        gate = self.gate(x)  # (B, N, C)
        
        # 投影到状态空间
        B_proj = self.B_proj(x)  # (B, N, d_state)
        C_proj = self.C_proj(x)  # (B, N, d_state)
        
        # 离散化参数 (C, d_state)
        delta = F.softplus(self.delta)
        
        # 双向扫描
        if self.bidirectional:
            # 前向扫描（从左到右，从上到下）
            y_fwd = self.ssm_scan_forward(B_proj, C_proj, delta, H, W)  # (B, N, d_state)
            
            # 后向扫描（从右到左，从下到上）
            y_bwd = self.ssm_scan_backward(B_proj, C_proj, delta, H, W)  # (B, N, d_state)
            
            # 拼接双向输出
            y = torch.cat([y_fwd, y_bwd], dim=-1)  # (B, N, 2*d_state)
        else:
            # 单向扫描
            y = self.ssm_scan_forward(B_proj, C_proj, delta, H, W)  # (B, N, d_state)
        
        # 先投影回 channels 维度，再应用门控
        y_projected = self.out_proj(y)  # (B, N, C)
        
        # 应用门控
        out = y_projected * gate
        
        return out
    
    def ssm_scan_forward(self, B, C, delta, H, W):
        """
        优化版前向扫描 - 使用向量化并行扫描算法
        
        核心思想：将递归公式转换为累积和形式
        state_t = A_bar * state_{t-1} + B_t
        => state_t = sum_{k=0}^{t}(A_bar^{t-k} * B_k)
        
        使用 cumsum 实现高效并行计算
        
        :param B: (B, N, d_state) - 输入投影
        :param C: (B, N, d_state) - 输出投影  
        :param delta: (C, d_state) - 离散化参数
        :param H, W: 空间维度
        :return: (B, N, d_state) - SSM 输出
        """
        B_batch, N, d_state = B.shape
        
        # 重塑为空间序列（按行优先顺序）
        B_flat = B.view(B_batch, N, d_state)
        C_flat = C.view(B_batch, N, d_state)
        
        # 计算离散的 A_bar（确保稳定性）
        delta_global = F.softplus(delta).mean(dim=0)  # (d_state,)
        A_mean = torch.tanh(self.A.mean(dim=0))  # (-1, 1)
        A_bar = torch.exp(-delta_global * (1 + A_mean.abs()))  # (d_state,), 确保 < 1
        
        # === 向量化并行扫描 ===
        # 方法：使用累积和代替循环
        # state_t = A_bar * state_{t-1} + B_t
        # 展开：state_t = B_0*A_bar^t + B_1*A_bar^{t-1} + ... + B_t
        
        # 预计算 A_bar 的幂次：A_powers[t] = A_bar^t
        t = torch.arange(N, device=B.device, dtype=B.dtype)  # (N,)
        A_powers = torch.pow(A_bar.unsqueeze(0), t.unsqueeze(1))  # (N, d_state)
        
        # 加权 B：B_weighted[t] = B_t * A_bar^{-t}
        A_powers_inv = 1.0 / (A_powers + 1e-8)  # 防止除零
        B_weighted = B_flat * A_powers_inv.unsqueeze(0)  # (B, N, d_state)
        
        # 累积和：cumsum[t] = sum_{k=0}^{t} B_weighted[k]
        B_cumsum = torch.cumsum(B_weighted, dim=1)  # (B, N, d_state)
        
        # 恢复状态：state_t = cumsum[t] * A_bar^t
        states = B_cumsum * A_powers.unsqueeze(0)  # (B, N, d_state)
        
        # 计算输出：output_t = C_t * state_t
        output = C_flat * states  # (B, N, d_state)
        
        return output
    
    def ssm_scan_backward(self, B, C, delta, H, W):
        """
        优化版后向扫描 - 使用向量化并行扫描算法
        
        从序列末尾开始，反向应用相同的逻辑
        
        :param B: (B, N, d_state)
        :param C: (B, N, d_state)
        :param delta: (C, d_state)
        :param H, W: 空间维度
        :return: (B, N, d_state)
        """
        B_batch, N, d_state = B.shape
        
        # 翻转序列（将后向扫描转换为前向扫描）
        B_flipped = torch.flip(B, dims=[1])  # (B, N, d_state)
        C_flipped = torch.flip(C, dims=[1])  # (B, N, d_state)
        
        # 计算离散的 A_bar
        delta_global = F.softplus(delta).mean(dim=0)  # (d_state,)
        A_mean = torch.tanh(self.A.mean(dim=0))  # (-1, 1)
        A_bar = torch.exp(-delta_global * (1 + A_mean.abs()))  # (d_state,)
        
        # === 向量化并行扫描（在翻转的序列上）===
        t = torch.arange(N, device=B.device, dtype=B.dtype)  # (N,)
        A_powers = torch.pow(A_bar.unsqueeze(0), t.unsqueeze(1))  # (N, d_state)
        
        A_powers_inv = 1.0 / (A_powers + 1e-8)
        B_weighted = B_flipped * A_powers_inv.unsqueeze(0)
        B_cumsum = torch.cumsum(B_weighted, dim=1)
        states = B_cumsum * A_powers.unsqueeze(0)
        output_flipped = C_flipped * states
        
        # 再次翻转回原始顺序
        output = torch.flip(output_flipped, dims=[1])  # (B, N, d_state)
        
        return output


class SimplifiedSSMBlock(nn.Module):
    """
    Simplified SSM Block with residual connections and MLP
    
    结构：LayerNorm -> SSM -> Residual -> LayerNorm -> MLP -> Residual
    """
    
    def __init__(self, channels, d_state=8, expansion_factor=4, drop_path=0.0):
        super().__init__()
        
        self.channels = channels
        self.d_state = d_state
        
        # 第一个归一化和 SSM 层
        self.norm1 = nn.LayerNorm(channels)
        self.ssm = SimplifiedSSM(channels, d_state, bidirectional=True)
        
        # 第二个归一化和 MLP 层
        self.norm2 = nn.LayerNorm(channels)
        expanded_channels = int(channels * expansion_factor)
        self.mlp = nn.Sequential(
            nn.Linear(channels, expanded_channels),
            nn.GELU(),
            nn.Linear(expanded_channels, channels)
        )
        
        # Stochastic Depth（可选）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x, H, W):
        """
        :param x: tensor with shape (B, N, C)
        :param H, W: spatial dimensions
        :return: tensor with shape (B, N, C)
        """
        # SSM 分支（带残差连接）
        x = x + self.drop_path(self.ssm(self.norm1(x), H, W))
        
        # MLP 分支（带残差连接）
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    
    在训练期间随机丢弃整个样本路径，提高泛化能力
    """
    
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # 计算保持概率
        keep_prob = 1 - self.drop_prob
        
        # 生成随机 mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        
        # 缩放输出
        output = x.div(keep_prob) * random_tensor
        
        return output


class OverlapPatchMerging(nn.Module):
    """
    重叠补丁合并层（从原有 ViTsubmodules.py 复用）
    
    使用卷积将图像分块并降维，执行 LayerNorm 归一化
    """
    
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, 
                            stride=stride, padding=padding)
        self.layerNorm = nn.LayerNorm(out_channels)
    
    def forward(self, patches):
        """
        Merge patches to reduce dimensions of input.
        
        :param patches: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, N, C) and spatial dimensions H, W
        """
        x = self.cn1(patches)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Flatten - (B,C,H*W); transpose B,HW, C
        x = self.layerNorm(x)
        return x, H, W  # B, N, EmbedDim
