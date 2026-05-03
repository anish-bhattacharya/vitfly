"""
分支 A: VMamba + LSTM 完整模型
用于无人机端到端避障任务

架构:
- VMamba 视觉编码器 (裁剪版)
- 特征融合层 (视觉特征 + 速度 + 四元数)
- LSTM 时序建模
- 输出层 (3维速度命令)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vmamba_encoder import VMambaEncoder, create_vmamba_encoder


class RefineInputs(nn.Module):
    """输入预处理模块"""
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        """
        X[0]: 深度图像 (B, 1, H, W)
        X[1]: 期望速度 (B, 3)
        X[2]: 当前四元数 (B, 4)
        """
        # 填充默认四元数 [1, 0, 0, 0]
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device)
            X[2][:, 0] = 1
            
        # 调整图像尺寸
        if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
            X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
            
        return X


class VMambaLSTMNet(nn.Module):
    """
    VMamba + LSTM 混合架构
    
    参数量:
    - VMamba Encoder: ~200K-2M (可配置)
    - LSTM: ~500K (input=519, hidden=128)
    - 输出层: ~2K
    - 总计: ~700K-2.5M
    """
    
    def __init__(
        self,
        vmamba_config=None,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        # 默认 VMamba 配置
        if vmamba_config is None:
            vmamba_config = {
                'in_channels': 1,
                'embed_dim': 64,
                'depth': 4,
                'd_state': 64,
                'dropout': dropout,
                'output_dim': 512
            }
        
        # VMamba 视觉编码器
        self.vmamba = create_vmamba_encoder(vmamba_config)
        
        # 输入预处理
        self.refine = RefineInputs()
        
        # 特征维度
        vmamba_output = vmamba_config.get('output_dim', 512)
        lstm_input = vmamba_output + 3 + 4  # 特征 + 速度 + 四元数
        
        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bias=False
        )
        
        # 输出层
        self.fc_out = nn.Linear(lstm_hidden, 3)
        
        # 保存配置
        self.vmamba_config = vmamba_config
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
    def forward(self, X, hidden_state=None):
        """
        X: [深度图像, 期望速度, 四元数]
        hidden_state: (h_0, c_0) 可选
        """
        # 预处理输入
        X = self.refine(X)
        
        # 视觉编码
        visual_feat = self.vmamba(X[0])  # (B, 512)
        
        # 特征融合: [视觉特征, 速度/10, 四元数]
        fused = torch.cat([
            visual_feat,
            X[1] * 0.1,  # 速度归一化
            X[2]
        ], dim=-1)  # (B, 519)
        
        # LSTM 时序建模
        if hidden_state is None:
            fused_seq = fused.unsqueeze(0)  # (1, B, 519)
            output, (h_n, c_n) = self.lstm(fused_seq)
            output = output.squeeze(0)  # (B, 3)
            hidden_state = (h_n, c_n)
        else:
            fused_seq = fused.unsqueeze(0)
            output, hidden_state = self.lstm(fused_seq, hidden_state)
            output = output.squeeze(0)
            
        # 输出速度命令
        output = self.fc_out(output)
        
        return output, hidden_state
    
    def get_parameter_count(self):
        """返回参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_vmamba_params(self):
        """返回 VMamba 部分参数量"""
        return sum(p.numel() for p in self.vmamba.parameters())
    
    def get_lstm_params(self):
        """返回 LSTM 部分参数量"""
        return sum(p.numel() for p in self.lstm.parameters())


def create_vmamba_lstm_model(config):
    """工厂函数: 根据配置创建模型"""
    vmamba_config = config.get('vmamba', {
        'in_channels': 1,
        'embed_dim': 64,
        'depth': 4,
        'd_state': 64,
        'dropout': 0.1,
        'output_dim': 512
    })
    
    return VMambaLSTMNet(
        vmamba_config=vmamba_config,
        lstm_hidden=config.get('lstm_hidden', 128),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.1)
    )


# 测试代码
if __name__ == '__main__':
    # 测试默认配置
    print("=" * 50)
    print("测试 VMamba + LSTM 模型")
    print("=" * 50)
    
    # 配置 1: 轻量版
    config_light = {
        'vmamba': {
            'embed_dim': 48,
            'depth': 2,
            'd_state': 8,
            'output_dim': 256
        },
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.1
    }
    
    # 配置 2: 标准版
    config_standard = {
        'vmamba': {
            'embed_dim': 64,
            'depth': 4,
            'd_state': 64,
            'output_dim': 512
        },
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.1
    }
    
    # 配置 3: 大容量版
    config_large = {
        'vmamba': {
            'embed_dim': 96,
            'depth': 6,
            'd_state': 32,
            'output_dim': 512
        },
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.1
    }
    
    configs = [
        ('轻量版', config_light),
        ('标准版', config_standard),
        ('大容量版', config_large)
    ]
    
    for name, config in configs:
        model = create_vmamba_lstm_model(config)
        total_params = model.get_parameter_count()
        vmamba_params = model.get_vmamba_params()
        lstm_params = model.get_lstm_params()
        
        print(f"\n{name}:")
        print(f"  VMamba 参数量: {vmamba_params:,}")
        print(f"  LSTM 参数量: {lstm_params:,}")
        print(f"  总参数量: {total_params:,}")
        
        # 测试前向传播
        x = torch.randn(1, 1, 60, 90)
        v = torch.randn(1, 3)
        q = torch.randn(1, 4)
        
        with torch.no_grad():
            output, _ = model([x, v, q])
            
        print(f"  输出形状: {output.shape}")
        
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
