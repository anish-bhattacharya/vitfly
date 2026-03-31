#!/usr/bin/env python3
"""
Mamba训练基础设施测试脚本
验证核心功能是否正常工作
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'training'))

def test_config_system():
    """测试配置系统"""
    print("测试配置系统...")
    
    from mamba_infra.config_manager import ConfigManager
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建配置管理器
        manager = ConfigManager()
        
        # 测试模板加载
        templates = manager.templates
        print(f"  加载模板数量: {len(templates)}")
        assert len(templates) >= 5, "应该至少有5个模板"
        
        # 测试配置创建
        config_path = manager.create_config(
            template_name='base_config',
            output_path=Path(temp_dir) / 'test_config.yaml',
            experiment_name='test_experiment',
            lr=1e-3
        )
        
        print(f"  创建配置: {config_path}")
        assert config_path.exists(), "配置文件应该存在"
        
        # 测试配置验证
        is_valid = manager.validate_config(config_path)
        print(f"  配置验证: {is_valid}")
        assert is_valid, "配置应该有效"
        
        print("  ✅ 配置系统测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置系统测试失败: {e}")
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_training_utils():
    """测试训练工具"""
    print("测试训练工具...")
    
    from mamba_infra.mamba_training_utils import MambaTrainingConfig
    
    try:
        # 测试配置类
        config = MambaTrainingConfig()
        
        # 检查默认配置
        assert 'device' in config, "配置应该包含device"
        assert 'lr' in config, "配置应该包含lr"
        assert 'N_eps' in config, "配置应该包含N_eps"
        
        # 测试配置修改
        config['lr'] = 2e-4
        assert config['lr'] == 2e-4, "配置修改应该生效"
        
        # 测试工作空间路径生成
        workspace = config.get_workspace_path()
        print(f"  工作空间路径: {workspace}")
        assert workspace is not None, "应该生成工作空间路径"
        
        print("  ✅ 训练工具测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 训练工具测试失败: {e}")
        return False

def test_validation_tools():
    """测试验证工具"""
    print("测试验证工具...")
    
    from mamba_infra.validation_tools import TrainingValidator
    
    # 创建临时工作空间
    temp_dir = tempfile.mkdtemp()
    workspace_dir = Path(temp_dir) / 'test_workspace'
    workspace_dir.mkdir(parents=True)
    
    try:
        # 创建模拟的训练历史
        history = {
            'train_metrics': [
                {'epoch': 1, 'train_loss': 0.5, 'timestamp': '2024-01-01T00:00:00'},
                {'epoch': 2, 'train_loss': 0.4, 'timestamp': '2024-01-01T00:10:00'},
                {'epoch': 3, 'train_loss': 0.3, 'val_loss': 0.35, 'timestamp': '2024-01-01T00:20:00'},
            ]
        }
        
        # 保存训练历史
        history_file = workspace_dir / 'training_history.json'
        import json
        with open(history_file, 'w') as f:
            json.dump(history, f)
        
        # 创建模拟配置
        config_file = workspace_dir / 'config.yaml'
        config = {
            'device': 'cpu',
            'lr': 1e-4,
            'N_eps': 10,
            'model_type': 'DroneMamba'
        }
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # 测试验证器
        validator = TrainingValidator(str(workspace_dir))
        validation_result = validator.validate_training_progress()
        
        print(f"  验证结果: {validation_result['status']}")
        assert 'status' in validation_result, "验证结果应该包含状态"
        
        print("  ✅ 验证工具测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 验证工具测试失败: {e}")
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_config_manager():
    """测试配置管理器"""
    print("测试配置管理器...")
    
    from mamba_infra.config_manager import ConfigManager, HyperparameterRange
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = ConfigManager()
        
        # 测试超参数搜索
        hyperparams = [
            HyperparameterRange('lr', [1e-4, 1e-3], '学习率'),
            HyperparameterRange('batch_size', [1, 2], '批大小')
        ]
        
        configs = manager.generate_hyperparameter_search(
            template_name='base_config',
            output_dir=Path(temp_dir) / 'search',
            hyperparameters=hyperparams
        )
        
        print(f"  生成超参数配置数量: {len(configs)}")
        assert len(configs) == 4, "应该生成4个配置 (2x2)"
        
        # 测试消融实验
        ablation_configs = manager.create_ablation_study(
            template_name='base_config',
            output_dir=Path(temp_dir) / 'ablation',
            ablation_params={
                'lr_decay': [True, False],
                'optimizer': ['adamw', 'sgd']
            }
        )
        
        print(f"  生成消融配置数量: {len(ablation_configs)}")
        assert len(ablation_configs) == 4, "应该生成4个消融配置"
        
        print("  ✅ 配置管理器测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置管理器测试失败: {e}")
        return False
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """主测试函数"""
    print("=" * 60)
    print("Mamba训练基础设施测试")
    print("=" * 60)
    
    tests = [
        test_config_system,
        test_training_utils,
        test_validation_tools,
        test_config_manager
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("⚠️  部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())