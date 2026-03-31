#!/usr/bin/env python3
"""
ViT-Fly Mamba 分支验证脚本
验证分支 B-E 的模型实现
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append('/root/vitfly')

def test_branch_B():
    """测试分支 B: MambaVision + SSM"""
    print("\n" + "="*60)
    print("测试分支 B: MambaVision + SSM")
    print("="*60)
    
    try:
        # 导入分支B模型
        sys.path.append('/root/vitfly/experiments/mamba_branches/branch_B_mambavision_ssm/models')
        from mambavision_ssm_model import create_mambavision_ssm_model
        
        # 创建模型
        config = {
            'mambavision_config': {
                'in_channels': 1,
                'stem_dim': 48,
                'stage_dims': (64, 128, 192),
                'depths': (2, 2, 2),
                'd_state': 12,
                'dropout': 0.1,
                'output_dim': 512
            },
            'ssm_d_state': 16,
            'ssm_hidden': 256,
            'ssm_layers': 2,
            'dropout': 0.1
        }
        
        model = create_mambavision_ssm_model(config)
        model.eval()
        
        # 测试输入 (ViT-Fly 格式)
        X = [
            torch.randn(2, 1, 60, 90),  # depth images
            torch.randn(2, 3),          # velocity
            torch.randn(2, 4)           # quaternion
        ]
        
        # 正向传播
        with torch.no_grad():
            output, hidden = model(X)
        
        # 检查输出形状
        assert output.shape == (2, 3), f"输出形状错误: {output.shape}"
        assert hidden.shape == (2, 32), f"隐藏状态形状错误: {hidden.shape}"
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ 模型实例化成功")
        print(f"✓ 正向传播成功")
        print(f"✓ 输出形状正确: {output.shape}")
        print(f"✓ 隐藏状态形状正确: {hidden.shape}")
        print(f"✓ 参数量: {params:,} ({params/1e6:.2f}M)")
        print(f"✓ 参数约束检查: {'通过' if params < 5e6 else '失败'} (<5M)")
        
        return {
            'branch': 'B',
            'status': 'success',
            'params': params,
            'output_shape': output.shape,
            'hidden_shape': hidden.shape
        }
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return {
            'branch': 'B',
            'status': 'failed',
            'error': str(e)
        }

def test_branch_C():
    """测试分支 C: CNN + Mamba-3"""
    print("\n" + "="*60)
    print("测试分支 C: CNN + Mamba-3")
    print("="*60)
    
    try:
        # 导入分支C模型
        sys.path.append('/root/vitfly/experiments/mamba_branches/branch_C_cnn_mamba3/models')
        from cnn_mamba3_model import create_cnn_mamba3_model
        
        # 创建模型
        config = {
            'cnn_config': {
                'in_channels': 1,
                'stem_dim': 32,
                'stage_dims': (32, 64, 128, 256),
                'output_dim': 512,
                'dropout': 0.1
            },
            'ssm_d_state': 32,
            'ssm_hidden': 256,
            'ssm_layers': 2,
            'dropout': 0.1
        }
        
        model = create_cnn_mamba3_model(config)
        model.eval()
        
        # 测试输入
        X = [
            torch.randn(2, 1, 60, 90),
            torch.randn(2, 3),
            torch.randn(2, 4)
        ]
        
        # 正向传播
        with torch.no_grad():
            output, hidden = model(X)
        
        # 检查输出形状
        assert output.shape == (2, 3), f"输出形状错误: {output.shape}"
        assert hidden.shape == (2, 64), f"隐藏状态形状错误: {hidden.shape}"
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ 模型实例化成功")
        print(f"✓ 正向传播成功")
        print(f"✓ 输出形状正确: {output.shape}")
        print(f"✓ 隐藏状态形状正确: {hidden.shape}")
        print(f"✓ 参数量: {params:,} ({params/1e6:.2f}M)")
        print(f"✓ 参数约束检查: {'通过' if params < 5e6 else '失败'} (<5M)")
        
        return {
            'branch': 'C',
            'status': 'success',
            'params': params,
            'output_shape': output.shape,
            'hidden_shape': hidden.shape
        }
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return {
            'branch': 'C',
            'status': 'failed',
            'error': str(e)
        }

def test_branch_D():
    """测试分支 D: STH-Mamba"""
    print("\n" + "="*60)
    print("测试分支 D: STH-Mamba")
    print("="*60)
    
    try:
        # 导入分支D模型
        sys.path.append('/root/vitfly/experiments/mamba_branches/branch_D_sth_mamba/models')
        from sth_mamba_model import create_sth_mamba_model
        
        # 创建模型
        config = {
            'spatial_dim': 256,
            'temporal_d_state': 16,
            'temporal_hidden': 256,
            'temporal_layers': 3,
            'dropout': 0.1
        }
        
        model = create_sth_mamba_model(config)
        model.eval()
        
        # 测试输入
        X = [
            torch.randn(2, 1, 60, 90),
            torch.randn(2, 3),
            torch.randn(2, 4)
        ]
        
        # 正向传播
        with torch.no_grad():
            output, state = model(X)
        
        # 检查输出形状
        assert output.shape == (2, 3), f"输出形状错误: {output.shape}"
        assert state.shape == (2, 16), f"状态形状错误: {state.shape}"
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ 模型实例化成功")
        print(f"✓ 正向传播成功")
        print(f"✓ 输出形状正确: {output.shape}")
        print(f"✓ 状态形状正确: {state.shape}")
        print(f"✓ 参数量: {params:,} ({params/1e6:.2f}M)")
        print(f"✓ 参数约束检查: {'通过' if params < 5e6 else '失败'} (<5M)")
        
        return {
            'branch': 'D',
            'status': 'success',
            'params': params,
            'output_shape': output.shape,
            'state_shape': state.shape
        }
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return {
            'branch': 'D',
            'status': 'failed',
            'error': str(e)
        }

def test_branch_E():
    """测试分支 E: DecisionMamba"""
    print("\n" + "="*60)
    print("测试分支 E: DecisionMamba")
    print("="*60)
    
    try:
        # 导入分支E模型
        sys.path.append('/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models')
        from decision_mamba_model import create_decision_mamba_model
        
        # 创建模型
        config = {
            'embed_dim': 192,
            'coarse_d_state': 16,
            'fine_d_state': 32,
            'num_patches': 15,
            'dropout': 0.1
        }
        
        model = create_decision_mamba_model(config)
        model.eval()
        
        # 测试输入
        X = [
            torch.randn(2, 1, 60, 90),
            torch.randn(2, 3),
            torch.randn(2, 4)
        ]
        
        # 正向传播
        with torch.no_grad():
            output, hidden = model(X)
        
        # 检查输出形状
        assert output.shape == (2, 3), f"输出形状错误: {output.shape}"
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ 模型实例化成功")
        print(f"✓ 正向传播成功")
        print(f"✓ 输出形状正确: {output.shape}")
        print(f"✓ 参数量: {params:,} ({params/1e6:.2f}M)")
        print(f"✓ 参数约束检查: {'通过' if params < 5e6 else '失败'} (<5M)")
        
        return {
            'branch': 'E',
            'status': 'success',
            'params': params,
            'output_shape': output.shape
        }
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return {
            'branch': 'E',
            'status': 'failed',
            'error': str(e)
        }

def test_input_compatibility():
    """测试输入格式兼容性"""
    print("\n" + "="*60)
    print("测试输入格式兼容性")
    print("="*60)
    
    test_cases = [
        {
            'name': '标准输入 (60x90)',
            'input': [
                torch.randn(2, 1, 60, 90),
                torch.randn(2, 3),
                torch.randn(2, 4)
            ]
        },
        {
            'name': '不同尺寸输入 (自动resize)',
            'input': [
                torch.randn(2, 1, 120, 180),
                torch.randn(2, 3),
                torch.randn(2, 4)
            ]
        },
        {
            'name': '缺失四元数 (自动填充)',
            'input': [
                torch.randn(2, 1, 60, 90),
                torch.randn(2, 3),
                None
            ]
        },
        {
            'name': '批量大小=1',
            'input': [
                torch.randn(1, 1, 60, 90),
                torch.randn(1, 3),
                torch.randn(1, 4)
            ]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        try:
            # 测试分支B作为代表
            sys.path.append('/root/vitfly/experiments/mamba_branches/branch_B_mambavision_ssm/models')
            from mambavision_ssm_model import create_mambavision_ssm_model
            
            model = create_mambavision_ssm_model({})
            model.eval()
            
            with torch.no_grad():
                output, hidden = model(test_case['input'])
            
            assert output.shape[0] == test_case['input'][0].shape[0]
            assert output.shape[1] == 3
            
            print(f"  ✓ 通过")
            results.append({
                'test': test_case['name'],
                'status': 'passed',
                'output_shape': output.shape
            })
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            results.append({
                'test': test_case['name'],
                'status': 'failed',
                'error': str(e)
            })
    
    return results

def generate_report(results):
    """生成验证报告"""
    print("\n" + "="*60)
    print("验证报告摘要")
    print("="*60)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'failed']
    
    print(f"\n总体结果:")
    print(f"  成功: {len(successful)}/{len(results)} 个分支")
    print(f"  失败: {len(failed)}/{len(results)} 个分支")
    
    if successful:
        print(f"\n成功分支详情:")
        for result in successful:
            print(f"  分支 {result['branch']}:")
            print(f"    - 参数量: {result.get('params', 0):,} ({result.get('params', 0)/1e6:.2f}M)")
            print(f"    - 输出形状: {result.get('output_shape', 'N/A')}")
            if 'hidden_shape' in result:
                print(f"    - 隐藏状态形状: {result.get('hidden_shape', 'N/A')}")
    
    if failed:
        print(f"\n失败分支详情:")
        for result in failed:
            print(f"  分支 {result['branch']}: {result.get('error', '未知错误')}")
    
    # 参数量统计
    param_counts = []
    for result in successful:
        if 'params' in result:
            param_counts.append((result['branch'], result['params']))
    
    if param_counts:
        print(f"\n参数量统计:")
        for branch, params in param_counts:
            print(f"  分支 {branch}: {params:,} ({params/1e6:.2f}M)")
        
        total_params = sum(p for _, p in param_counts)
        avg_params = total_params / len(param_counts) if param_counts else 0
        print(f"\n  平均参数量: {avg_params:,.0f} ({avg_params/1e6:.2f}M)")
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return {
        'total_branches': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'param_counts': param_counts
    }

def main():
    """主函数"""
    print("ViT-Fly Mamba 分支验证")
    print("="*60)
    
    # 测试所有分支
    results = []
    results.append(test_branch_B())
    results.append(test_branch_C())
    results.append(test_branch_D())
    results.append(test_branch_E())
    
    # 测试输入兼容性
    compatibility_results = test_input_compatibility()
    
    # 生成报告
    report = generate_report(results)
    
    # 总结
    print("\n" + "="*60)
    print("验证完成")
    print("="*60)
    
    if report['successful'] == report['total_branches']:
        print("✅ 所有分支验证通过!")
    else:
        print(f"⚠️  {report['failed']} 个分支验证失败")
    
    # 检查参数量约束
    all_within_limit = all(p < 5e6 for _, p in report.get('param_counts', []))
    if all_within_limit:
        print("✅ 所有分支参数量 < 5M 约束")
    else:
        print("❌ 部分分支参数量超过 5M 约束")
    
    return report

if __name__ == '__main__':
    main()