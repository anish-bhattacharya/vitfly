#!/usr/bin/env python3
"""
Mamba训练基础设施完整工作流程示例
演示从配置创建到训练评估的完整流程
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

def demonstrate_complete_workflow():
    """演示完整工作流程"""
    print("=" * 60)
    print("Mamba训练基础设施完整工作流程演示")
    print("=" * 60)
    
    # 创建临时目录用于演示
    temp_dir = Path(tempfile.mkdtemp())
    print(f"临时目录: {temp_dir}")
    
    try:
        # 1. 配置管理
        print("\n1. 配置管理")
        print("-" * 40)
        
        from mamba_infra.config_manager import ConfigManager, HyperparameterRange
        
        manager = ConfigManager()
        
        # 创建基础配置
        base_config_path = manager.create_config(
            template_name='base_config',
            output_path=temp_dir / 'base_config.yaml',
            experiment_name='demo_experiment',
            device='cpu',  # 使用CPU进行演示
            N_eps=5,  # 只训练5个epoch用于演示
            val_freq=2,
            save_model_freq=3
        )
        print(f"  创建基础配置: {base_config_path}")
        
        # 2. 超参数搜索
        print("\n2. 超参数搜索")
        print("-" * 40)
        
        hyperparams = [
            HyperparameterRange('lr', [1e-4, 5e-4], '学习率'),
            HyperparameterRange('batch_size', [1], '批大小')
        ]
        
        search_configs = manager.generate_hyperparameter_search(
            template_name='base_config',
            output_dir=temp_dir / 'hyperparam_search',
            hyperparameters=hyperparams
        )
        print(f"  生成超参数配置: {len(search_configs)} 个")
        
        # 3. 创建训练脚本
        print("\n3. 创建批量训练脚本")
        print("-" * 40)
        
        batch_script = manager.generate_training_script(
            config_paths=search_configs,
            output_script=temp_dir / 'batch_train.sh',
            gpu_ids=[0]  # 使用GPU 0
        )
        print(f"  生成批量训练脚本: {batch_script}")
        
        # 4. 训练配置验证
        print("\n4. 训练配置验证")
        print("-" * 40)
        
        from mamba_infra.mamba_training_utils import MambaTrainingConfig
        
        config = MambaTrainingConfig(str(base_config_path))
        print(f"  配置验证通过:")
        print(f"    - 设备: {config['device']}")
        print(f"    - 学习率: {config['lr']}")
        print(f"    - 训练epoch: {config['N_eps']}")
        print(f"    - 模型类型: {config['model_type']}")
        
        # 5. 模拟训练环境设置
        print("\n5. 模拟训练环境设置")
        print("-" * 40)
        
        from mamba_infra.mamba_training_utils import setup_training_environment
        
        env = setup_training_environment(config)
        print(f"  训练环境组件:")
        print(f"    - 工作空间: {env['workspace'].name}")
        print(f"    - 损失函数: {type(env['loss_fn']).__name__}")
        print(f"    - 检查点管理器: {type(env['checkpoint_manager']).__name__}")
        print(f"    - 训练监控器: {type(env['monitor']).__name__}")
        
        # 6. 验证工具演示
        print("\n6. 验证工具演示")
        print("-" * 40)
        
        from mamba_infra.validation_tools import TrainingValidator
        
        # 创建模拟的训练历史
        demo_workspace = temp_dir / 'demo_workspace'
        demo_workspace.mkdir(parents=True)
        
        # 保存模拟配置
        import yaml
        with open(demo_workspace / 'config.yaml', 'w') as f:
            yaml.dump(config.config, f)
        
        # 创建模拟训练历史
        import json
        history = {
            'train_metrics': [
                {'epoch': 1, 'train_loss': 0.5, 'learning_rate': 1e-4, 'timestamp': '2024-01-01T00:00:00'},
                {'epoch': 2, 'train_loss': 0.4, 'learning_rate': 1e-4, 'timestamp': '2024-01-01T00:10:00'},
                {'epoch': 3, 'train_loss': 0.35, 'val_loss': 0.38, 'learning_rate': 1e-4, 'timestamp': '2024-01-01T00:20:00'},
                {'epoch': 4, 'train_loss': 0.3, 'learning_rate': 9.5e-5, 'timestamp': '2024-01-01T00:30:00'},
                {'epoch': 5, 'train_loss': 0.28, 'val_loss': 0.32, 'learning_rate': 9.5e-5, 'timestamp': '2024-01-01T00:40:00'},
            ]
        }
        
        with open(demo_workspace / 'training_history.json', 'w') as f:
            json.dump(history, f)
        
        # 验证训练进度
        validator = TrainingValidator(str(demo_workspace))
        validation = validator.validate_training_progress()
        print(f"  训练验证结果: {validation['status']}")
        print(f"  最终训练损失: {validation['final_train_loss']:.4f}")
        print(f"  最终验证损失: {validation['final_val_loss']:.4f}")
        
        if validation['issues']:
            print(f"  检测到问题: {validation['issues']}")
        else:
            print("  未检测到明显问题")
        
        # 7. 配置比较演示
        print("\n7. 配置比较演示")
        print("-" * 40)
        
        differences = manager.compare_configs([str(c) for c in search_configs[:2]])
        print(f"  配置差异数量: {len(differences)}")
        
        if differences:
            print("  主要差异:")
            for param, values in list(differences.items())[:3]:  # 只显示前3个
                print(f"    - {param}: {values}")
        
        # 8. 结果分析演示
        print("\n8. 结果分析演示")
        print("-" * 40)
        
        from mamba_infra.validation_tools import ResultAnalyzer
        
        # 创建模拟结果
        results_dir = temp_dir / 'results'
        results_dir.mkdir(parents=True)
        
        # 创建几个模拟的评估结果
        for i, config_file in enumerate(search_configs[:2]):
            exp_dir = results_dir / f'experiment_{i}'
            exp_dir.mkdir(parents=True)
            
            eval_data = {
                'config': {
                    'branch': 'B',
                    'lr': 1e-4 if i == 0 else 5e-4,
                    'batch_size': 1
                },
                'speed_metrics': {
                    'inference_time_ms': 2.5 + i * 0.5,
                    'fps': 400 - i * 50,
                    'total_parameters': 3300000
                },
                'performance_metrics': {
                    'loss': 0.3 - i * 0.05,
                    'mae': 0.25 - i * 0.03,
                    'rmse': 0.35 - i * 0.04
                },
                'checkpoint_info': {
                    'epoch': 5,
                    'metrics': {'val_loss': 0.32 - i * 0.06}
                }
            }
            
            with open(exp_dir / 'evaluation_metrics.json', 'w') as f:
                json.dump(eval_data, f, indent=2)
        
        # 分析结果
        analyzer = ResultAnalyzer(str(results_dir))
        results_df = analyzer.analyze_multiple_experiments()
        
        if not results_df.empty:
            print(f"  分析实验数量: {len(results_df)}")
            print("  最佳验证损失:")
            best_experiment = results_df.nsmallest(1, 'val_loss').iloc[0]
            print(f"    - 实验: {best_experiment['experiment']}")
            print(f"    - 学习率: {best_experiment['lr']}")
            print(f"    - 验证损失: {best_experiment['val_loss']:.4f}")
            print(f"    - 推理时间: {best_experiment['inference_time_ms']:.2f} ms")
        
        # 9. 使用文档生成
        print("\n9. 使用示例")
        print("-" * 40)
        
        print("""
  快速开始命令:
  
  # 训练分支B
  python training/mamba_infra/train_mamba_branch.py \\
    --config training/mamba_infra/config_templates/branch_B_config.yaml
  
  # 训练分支C并覆盖参数
  python training/mamba_infra/train_mamba_branch.py \\
    --config training/mamba_infra/config_templates/branch_C_config.yaml \\
    --lr 1.5e-4 \\
    --epochs 100 \\
    --experiment_name my_custom_experiment
  
  # 监控训练
  tensorboard --logdir training/logs
  
  # 评估模型
  python -c "
  from mamba_infra.validation_tools import ModelEvaluator
  evaluator = ModelEvaluator('training/logs/my_experiment/checkpoints/checkpoint_best.pt')
  metrics = evaluator.evaluate_inference_speed()
  print(f'推理时间: {metrics[\"inference_time_ms\"]:.2f} ms')
  "
        """)
        
        print("\n" + "=" * 60)
        print("工作流程演示完成!")
        print("=" * 60)
        print(f"\n所有演示文件保存在: {temp_dir}")
        print("查看详细文档: training/mamba_infra/README.md")
        
        return True
        
    except Exception as e:
        print(f"\n演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 询问是否清理临时目录
        keep_files = input(f"\n是否保留临时文件? (y/n, 目录: {temp_dir}): ").lower().strip()
        if keep_files != 'y':
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("已清理临时文件")
        else:
            print(f"临时文件保留在: {temp_dir}")

if __name__ == '__main__':
    demonstrate_complete_workflow()