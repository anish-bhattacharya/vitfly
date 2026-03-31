#!/usr/bin/env python3
"""
ViT-Fly Mamba分支B-E 数据准备脚本
整合数据验证、分析和准备功能
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ViT-Fly 数据准备工具")
    parser.add_argument("--action", choices=["validate", "analyze", "prepare", "all"], 
                       default="all", help="执行的操作")
    parser.add_argument("--data-dir", default="training/datasets/data",
                       help="数据集目录路径")
    parser.add_argument("--output-dir", default="training/datasets/reports",
                       help="报告输出目录")
    parser.add_argument("--create-sample", action="store_true",
                       help="创建示例数据集")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("ViT-Fly Mamba分支B-E 数据准备工具")
    print("=" * 60)
    
    # 如果需要创建示例数据
    if args.create_sample:
        print("\n创建示例数据集...")
        from create_sample_data import create_sample_dataset
        create_sample_dataset()
    
    # 执行请求的操作
    if args.action in ["validate", "all"]:
        print("\n执行数据验证...")
        from check_dataset import main as validate_main
        # 修改sys.argv以传递参数
        sys.argv = ["check_dataset.py"]
        validate_main()
    
    if args.action in ["analyze", "all"]:
        print("\n执行数据分析...")
        from analyze_dataset import main as analyze_main
        sys.argv = ["analyze_dataset.py"]
        analyze_main()
    
    if args.action in ["prepare", "all"]:
        print("\n准备训练数据...")
        prepare_training_data(args.data_dir)
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)

def prepare_training_data(data_dir: str):
    """准备训练数据"""
    data_path = Path(data_dir)
    
    print(f"准备训练数据从: {data_path}")
    
    # 检查数据目录
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        print("请从以下位置下载数据:")
        print("  URL: https://upenn.app.box.com/v/ViT-quad-datashare")
        print("  密码: vitfly2025")
        print("  文件: data.zip (2.5GB)")
        print("  解压命令: unzip data.zip -d training/datasets/data")
        return
    
    # 检查是否有数据
    trajectory_folders = [f for f in data_path.iterdir() if f.is_dir()]
    if not trajectory_folders:
        print("警告: 数据目录为空")
        print("请下载数据集或使用 --create-sample 创建示例数据")
        return
    
    print(f"找到 {len(trajectory_folders)} 个轨迹文件夹")
    
    # 创建训练准备报告
    report_path = Path("training/datasets/reports/training_preparation_report.txt")
    with open(report_path, 'w') as f:
        f.write("ViT-Fly 训练数据准备报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据目录: {data_path}\n")
        f.write(f"轨迹数量: {len(trajectory_folders)}\n\n")
        
        # 统计每个轨迹的信息
        f.write("轨迹统计:\n")
        f.write("-" * 40 + "\n")
        
        total_images = 0
        for traj_folder in trajectory_folders[:10]:  # 只显示前10个
            image_files = list(traj_folder.glob("*.png"))
            csv_file = traj_folder / "data.csv"
            
            has_csv = csv_file.exists()
            image_count = len(image_files)
            total_images += image_count
            
            f.write(f"{traj_folder.name}: {image_count} 图像, CSV: {'有' if has_csv else '无'}\n")
        
        if len(trajectory_folders) > 10:
            f.write(f"... 还有 {len(trajectory_folders) - 10} 个轨迹\n")
        
        f.write(f"\n总图像数: {total_images}\n")
        
        # 训练配置建议
        f.write("\n训练配置建议:\n")
        f.write("-" * 40 + "\n")
        
        if total_images > 10000:
            f.write("1. 数据集规模: 大型 (适合完整训练)\n")
            f.write("2. 建议批次大小: 32-64\n")
            f.write("3. 建议训练轮数: 100-200\n")
        elif total_images > 1000:
            f.write("1. 数据集规模: 中型 (适合训练和验证)\n")
            f.write("2. 建议批次大小: 16-32\n")
            f.write("3. 建议训练轮数: 50-100\n")
        else:
            f.write("1. 数据集规模: 小型 (适合测试和调试)\n")
            f.write("2. 建议批次大小: 8-16\n")
            f.write("3. 建议训练轮数: 20-50\n")
        
        # Mamba分支特定建议
        f.write("\nMamba分支B-E特定建议:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 输入尺寸: 60×90 深度图像\n")
        f.write("2. 输出: 线性速度命令 (v_x, v_y, v_z)\n")
        f.write("3. 建议使用ViTLSTM或DroneMamba模型\n")
        f.write("4. 注意数据归一化:\n")
        f.write("   - 图像: 除以255.0\n")
        f.write("   - 速度: 根据统计数据归一化\n")
        
        # 下一步操作
        f.write("\n下一步操作:\n")
        f.write("-" * 40 + "\n")
        f.write("1. 运行训练: python training/train.py --config training/config/train.txt\n")
        f.write("2. 监控训练: tensorboard --logdir training/logs\n")
        f.write("3. 测试模型: bash launch_evaluation.bash 1 vision\n")
    
    print(f"训练准备报告已保存到: {report_path}")
    
    # 创建训练配置文件模板
    config_template = """# ViT-Fly Mamba分支B-E 训练配置
# 生成时间: {timestamp}

device = cuda
basedir = /root/catkin_ws/src/vitfly
logdir = training/logs
datadir = training/datasets

dataset = data
short = 0
val_split = 0.2

# 模型选择: ConvNet, LSTMnet, UNet, ViT, ViTLSTM, DroneMamba
model_type = ViTLSTM
load_checkpoint = False
checkpoint_path = ''

# 训练参数
lr = 1e-4
N_eps = 100
lr_warmup_epochs = 5
lr_decay = False
save_model_freq = 25
val_freq = 10

# Mamba分支特定参数
# 注意: DroneMamba模型需要额外的配置
# mamba_config = models/mamba_config.json

# 数据增强 (可选)
# data_augmentation = True
# augment_rotation = 5.0  # 度
# augment_translation = 0.1  # 比例
# augment_brightness = 0.2  # 比例
"""
    
    config_path = Path("training/config/train_mamba_branch.txt")
    from datetime import datetime
    config_content = config_template.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"训练配置模板已保存到: {config_path}")
    print("\n要开始训练，运行:")
    print(f"  python training/train.py --config {config_path}")

if __name__ == "__main__":
    main()