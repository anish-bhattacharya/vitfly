"""
分支 A: 张量形状验证测试
验证所有内部张量的 shape 和 size 是否匹配
"""

import torch
import torch.nn as nn
from vmamba_encoder import VMambaEncoder, SS2D, VSSBlock, PatchEmbedding
from vmamba_lstm_model import VMambaLSTMNet


def test_patch_embedding():
    """测试 Patch Embedding 层"""
    print("=" * 60)
    print("测试 Patch Embedding")
    print("=" * 60)
    
    x = torch.randn(2, 1, 60, 90)  # (B, C, H, W)
    print(f"输入形状：{x.shape}")
    
    pe = PatchEmbedding(in_channels=1, embed_dim=64, patch_size=4)
    out = pe(x)
    
    # 预期输出：(B, H/4, W/4, C) = (2, 15, 22, 64)
    expected_shape = (2, 15, 22, 64)
    print(f"输出形状：{out.shape}")
    print(f"预期形状：{expected_shape}")
    assert out.shape == expected_shape, f"形状不匹配！{out.shape} != {expected_shape}"
    print("✅ Patch Embedding 测试通过\n")
    return True


def test_ss2d():
    """测试 SS2D 模块"""
    print("=" * 60)
    print("测试 SS2D 模块")
    print("=" * 60)
    
    B, H, W, C = 2, 15, 22, 64
    x = torch.randn(B, H, W, C)
    print(f"输入形状：{x.shape}")
    
    ss2d = SS2D(dim=C, d_state=16)
    out = ss2d(x)
    
    # 预期输出：与输入相同 (B, H, W, C)
    expected_shape = (B, H, W, C)
    print(f"输出形状：{out.shape}")
    print(f"预期形状：{expected_shape}")
    assert out.shape == expected_shape, f"形状不匹配！{out.shape} != {expected_shape}"
    print("✅ SS2D 测试通过\n")
    return True


def test_vss_block():
    """测试 VSS Block"""
    print("=" * 60)
    print("测试 VSS Block")
    print("=" * 60)
    
    B, H, W, C = 2, 15, 22, 64
    x = torch.randn(B, H, W, C)
    print(f"输入形状：{x.shape}")
    
    block = VSSBlock(dim=C, d_state=16, dropout=0.1)
    out = block(x)
    
    # 预期输出：与输入相同 (B, H, W, C)
    expected_shape = (B, H, W, C)
    print(f"输出形状：{out.shape}")
    print(f"预期形状：{expected_shape}")
    assert out.shape == expected_shape, f"形状不匹配！{out.shape} != {expected_shape}"
    print("✅ VSS Block 测试通过\n")
    return True


def test_vmamba_encoder():
    """测试 VMamba 编码器"""
    print("=" * 60)
    print("测试 VMamba 编码器")
    print("=" * 60)
    
    configs = [
        {'embed_dim': 48, 'depth': 2, 'd_state': 8, 'output_dim': 256},
        {'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512},
        {'embed_dim': 96, 'depth': 6, 'd_state': 32, 'output_dim': 512},
    ]
    
    for cfg in configs:
        print(f"\n配置：dim={cfg['embed_dim']}, depth={cfg['depth']}, d_state={cfg['d_state']}")
        
        encoder = VMambaEncoder(
            in_channels=1,
            embed_dim=cfg['embed_dim'],
            depth=cfg['depth'],
            d_state=cfg['d_state'],
            dropout=0.1,
            output_dim=cfg['output_dim']
        )
        
        x = torch.randn(2, 1, 60, 90)
        print(f"  输入形状：{x.shape}")
        
        out = encoder(x)
        expected_shape = (2, cfg['output_dim'])
        
        print(f"  输出形状：{out.shape}")
        print(f"  预期形状：{expected_shape}")
        assert out.shape == expected_shape, f"形状不匹配！{out.shape} != {expected_shape}"
        
        # 验证参数量
        params = encoder.get_parameter_count()
        print(f"  参数量：{params:,}")
    
    print("\n✅ VMamba 编码器测试通过\n")
    return True


def test_feature_fusion():
    """测试特征融合层维度"""
    print("=" * 60)
    print("测试特征融合")
    print("=" * 60)
    
    B = 2
    visual_feat = torch.randn(B, 512)  # VMamba 输出
    des_vel = torch.randn(B, 3) * 0.1  # 速度归一化
    quat = torch.randn(B, 4)  # 四元数
    
    print(f"视觉特征：{visual_feat.shape}")
    print(f"速度 (归一化): {des_vel.shape}")
    print(f"四元数：{quat.shape}")
    
    # 融合
    fused = torch.cat([visual_feat, des_vel, quat], dim=-1)
    print(f"融合后：{fused.shape}")
    
    expected_dim = 512 + 3 + 4  # 519
    assert fused.shape[-1] == expected_dim, f"融合维度错误！{fused.shape[-1]} != {expected_dim}"
    print(f"✅ 特征融合测试通过 (总维度={expected_dim})\n")
    return True


def test_lstm_input_output():
    """测试 LSTM 输入输出维度"""
    print("=" * 60)
    print("测试 LSTM 输入输出")
    print("=" * 60)
    
    B = 2
    lstm_input_dim = 519  # 512 + 3 + 4
    lstm_hidden = 128
    lstm_layers = 2
    
    # 创建 LSTM
    lstm = nn.LSTM(
        input_size=lstm_input_dim,
        hidden_size=lstm_hidden,
        num_layers=lstm_layers,
        dropout=0.1,
        bias=False
    )
    
    # LSTM 输入：(seq_len, B, input_dim)
    x = torch.randn(1, B, lstm_input_dim)
    print(f"LSTM 输入形状：{x.shape}")
    
    output, (h_n, c_n) = lstm(x)
    
    # LSTM 输出：(seq_len, B, hidden_size)
    print(f"LSTM 输出形状：{output.shape}")
    print(f"隐藏状态形状：h_n={h_n.shape}, c_n={c_n.shape}")
    
    assert output.shape == (1, B, lstm_hidden), f"LSTM 输出形状错误！"
    assert h_n.shape == (lstm_layers, B, lstm_hidden), f"h_n 形状错误！"
    assert c_n.shape == (lstm_layers, B, lstm_hidden), f"c_n 形状错误！"
    
    # 输出层
    fc_out = nn.Linear(lstm_hidden, 3)
    output_squeezed = output.squeeze(0)  # (B, hidden)
    print(f"squeeze 后：{output_squeezed.shape}")
    
    final_out = fc_out(output_squeezed)
    print(f"最终输出：{final_out.shape}")
    assert final_out.shape == (B, 3), f"最终输出形状错误！"
    
    print("✅ LSTM 输入输出测试通过\n")
    return True


def test_full_model():
    """测试完整模型"""
    print("=" * 60)
    print("测试完整模型 VMamba+LSTM")
    print("=" * 60)
    
    configs = [
        ('轻量版', {'vmamba': {'embed_dim': 48, 'depth': 2, 'd_state': 8, 'output_dim': 256}}),
        ('标准版', {'vmamba': {'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512}}),
        ('大容量版', {'vmamba': {'embed_dim': 96, 'depth': 6, 'd_state': 32, 'output_dim': 512}}),
    ]
    
    for name, cfg in configs:
        print(f"\n{name}:")
        
        model = VMambaLSTMNet(
            vmamba_config=cfg['vmamba'],
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.1
        )
        
        # 输入
        depth_img = torch.randn(2, 1, 60, 90)
        des_vel = torch.randn(2, 3)
        quat = torch.randn(2, 4)
        
        print(f"  输入：img={depth_img.shape}, vel={des_vel.shape}, quat={quat.shape}")
        
        # 前向传播
        with torch.no_grad():
            output, hidden = model([depth_img, des_vel, quat])
        
        print(f"  输出：{output.shape}")
        assert output.shape == (2, 3), f"输出形状错误！{output.shape} != (2, 3)"
        
        # 参数量验证
        total_params = model.get_parameter_count()
        vmamba_params = model.get_vmamba_params()
        lstm_params = model.get_lstm_params()
        
        print(f"  参数：VMamba={vmamba_params:,}, LSTM={lstm_params:,}, 总计={total_params:,}")
    
    print("\n✅ 完整模型测试通过\n")
    return True


def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 60)
    print("测试梯度流动")
    print("=" * 60)
    
    model = VMambaLSTMNet(
        vmamba_config={'embed_dim': 64, 'depth': 4, 'd_state': 16, 'output_dim': 512},
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.1
    )
    model.train()
    
    # 输入
    depth_img = torch.randn(2, 1, 60, 90, requires_grad=True)
    des_vel = torch.randn(2, 3, requires_grad=True)
    quat = torch.randn(2, 4, requires_grad=True)
    
    # 前向传播
    output, _ = model([depth_img, des_vel, quat])
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    
    # 检查梯度
    has_grad = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad.append(name)
            if param.grad.isnan().any():
                print(f"  ⚠️  {name} 梯度包含 NaN!")
            if param.grad.isinf().any():
                print(f"  ⚠️  {name} 梯度包含 Inf!")
    
    print(f"  有梯度的参数：{len(has_grad)}/{len(list(model.parameters()))}")
    
    # 检查输入梯度
    assert depth_img.grad is not None, "深度图输入无梯度!"
    assert des_vel.grad is not None, "速度输入无梯度!"
    assert quat.grad is not None, "四元数输入无梯度!"
    
    print("✅ 梯度流动测试通过\n")
    return True


if __name__ == '__main__':
    print("\n" + "🔍" * 30)
    print("分支 A: 张量形状验证测试套件")
    print("🔍" * 30 + "\n")
    
    tests = [
        ("Patch Embedding", test_patch_embedding),
        ("SS2D 模块", test_ss2d),
        ("VSS Block", test_vss_block),
        ("VMamba 编码器", test_vmamba_encoder),
        ("特征融合", test_feature_fusion),
        ("LSTM 输入输出", test_lstm_input_output),
        ("完整模型", test_full_model),
        ("梯度流动", test_gradient_flow),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} 测试失败：{e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"测试结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ 所有张量形状验证通过！")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查！")
