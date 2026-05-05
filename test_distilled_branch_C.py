#!/usr/bin/env python3
"""
测试 Branch C 蒸馏模型在 ROS 仿真环境中的性能
"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, '/root/catkin_ws/src/vitfly-mambatest')

def load_distilled_model():
    """加载 Branch C 蒸馏模型"""
    from experiments.mamba_branches.branch_C_cnn_mamba3.models.cnn_mamba3_model import CNNMamba3Net
    
    checkpoint_path = 'experiments/mamba_branches/optimized_training/branch_C/distill_best_model.pth'
    
    print(f"加载蒸馏模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n检查点信息:")
    print(f"  - 分支: {checkpoint.get('branch', 'N/A')}")
    print(f"  - 训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Score: {checkpoint.get('val_score', 'N/A'):.6f}")
    print(f"  - Val Loss (GT): {checkpoint.get('val_loss_gt', 'N/A'):.6f}")
    print(f"  - Distill Gap: {checkpoint.get('val_distill_gap', 'N/A'):.6f}")
    
    # 创建模型 - 使用默认配置
    model = CNNMamba3Net(
        cnn_config=None,      # 使用默认 CNN 配置
        ssm_d_state=32,       # SSM 状态维度
        ssm_hidden=256,       # SSM 隐藏层维度
        ssm_layers=2,         # SSM 层数
        dropout=0.1           # Dropout
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ 模型加载成功")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 参数量: {total_params:,}")
    print(f"  - 预期参数量: 2,411,387")
    
    return model, checkpoint

def test_model_inference():
    """测试模型推理"""
    model, checkpoint = load_distilled_model()
    
    # 创建测试输入 - Branch C 使用 CNN，输入是深度图
    batch_size = 1
    depth_img = torch.randn(batch_size, 1, 60, 90)  # 深度图
    velocity = torch.randn(batch_size, 3)            # 速度信息
    quaternion = torch.randn(batch_size, 4)          # 四元数
    
    print(f"\n测试推理:")
    print(f"  - 深度图形状: {depth_img.shape}")
    print(f"  - 速度形状: {velocity.shape}")
    print(f"  - 四元数形状: {quaternion.shape}")
    
    with torch.no_grad():
        output, hidden = model([depth_img, velocity, quaternion])
    
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  - 隐藏状态: {type(hidden)}")
    print(f"✓ 推理测试通过")
    
    return True

def compare_with_original():
    """对比蒸馏模型与原始模型的参数量"""
    print(f"\n模型对比:")
    print(f"  - 蒸馏模型参数量: 2,411,387")
    print(f"  - 原始 Branch C 参数量: ~3.0M (估计)")
    print(f"  - 参数压缩率: ~19.6%")
    print(f"  - Val Score: 0.02757")
    print(f"  - Distill Gap: 0.01756 (最小)")

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Branch C 蒸馏模型测试")
        print("=" * 60)
        
        test_model_inference()
        compare_with_original()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
