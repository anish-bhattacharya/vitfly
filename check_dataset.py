#!/usr/bin/env python3
"""
ViT-Fly 数据集验证脚本
用于验证Mamba分支B-E训练数据的格式和完整性

验证内容：
1. 数据集目录结构
2. 图像文件格式（60×90 PNG）
3. CSV数据格式
4. 数据完整性（图像与CSV对应关系）
5. 基本统计信息
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class DatasetValidator:
    def __init__(self, data_dir: str = "training/datasets/data"):
        """
        初始化数据集验证器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = Path(data_dir)
        self.results = {
            "total_trajectories": 0,
            "valid_trajectories": 0,
            "total_images": 0,
            "image_errors": [],
            "csv_errors": [],
            "mismatch_errors": [],
            "statistics": {}
        }
        
    def validate_dataset(self) -> Dict:
        """
        验证整个数据集
        
        Returns:
            验证结果字典
        """
        print(f"验证数据集目录: {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"错误: 数据集目录不存在: {self.data_dir}")
            self.results["error"] = f"数据集目录不存在: {self.data_dir}"
            return self.results
            
        # 获取所有轨迹文件夹
        trajectory_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        self.results["total_trajectories"] = len(trajectory_folders)
        
        print(f"找到 {len(trajectory_folders)} 个轨迹文件夹")
        
        for traj_folder in trajectory_folders:
            print(f"\n验证轨迹: {traj_folder.name}")
            is_valid = self._validate_trajectory(traj_folder)
            
            if is_valid:
                self.results["valid_trajectories"] += 1
                
        # 生成统计报告
        self._generate_statistics()
        
        return self.results
    
    def _validate_trajectory(self, traj_folder: Path) -> bool:
        """
        验证单个轨迹文件夹
        
        Args:
            traj_folder: 轨迹文件夹路径
            
        Returns:
            是否有效
        """
        is_valid = True
        
        # 1. 检查图像文件
        image_files = sorted(traj_folder.glob("*.png"))
        if not image_files:
            print(f"  警告: 轨迹 {traj_folder.name} 中没有PNG图像文件")
            self.results["image_errors"].append(f"{traj_folder.name}: 无PNG文件")
            is_valid = False
        else:
            # 验证每个图像
            for img_file in image_files:
                img_valid, error_msg = self._validate_image(img_file)
                if not img_valid:
                    self.results["image_errors"].append(f"{traj_folder.name}/{img_file.name}: {error_msg}")
                    is_valid = False
                    
            self.results["total_images"] += len(image_files)
            
        # 2. 检查CSV文件
        csv_file = traj_folder / "data.csv"
        if not csv_file.exists():
            print(f"  错误: 轨迹 {traj_folder.name} 中没有data.csv文件")
            self.results["csv_errors"].append(f"{traj_folder.name}: 无data.csv文件")
            return False
            
        # 验证CSV格式
        csv_valid, csv_data, error_msg = self._validate_csv(csv_file)
        if not csv_valid:
            self.results["csv_errors"].append(f"{traj_folder.name}: {error_msg}")
            is_valid = False
            
        # 3. 检查图像和CSV数据对应关系
        if image_files and csv_data is not None:
            mismatch_valid, error_msg = self._validate_image_csv_match(image_files, csv_data)
            if not mismatch_valid:
                self.results["mismatch_errors"].append(f"{traj_folder.name}: {error_msg}")
                is_valid = False
                
        return is_valid
    
    def _validate_image(self, img_path: Path) -> Tuple[bool, Optional[str]]:
        """
        验证单个图像文件
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 读取图像
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "无法读取图像文件"
                
            # 检查尺寸 (根据dataloading.py应该是60×90)
            height, width = img.shape
            if height != 60 or width != 90:
                return False, f"图像尺寸错误: {height}×{width} (应为60×90)"
                
            # 检查数据类型
            if img.dtype != np.uint8:
                return False, f"图像数据类型错误: {img.dtype} (应为uint8)"
                
            # 检查值范围
            if img.min() < 0 or img.max() > 255:
                return False, f"图像值范围错误: [{img.min()}, {img.max()}] (应为[0, 255])"
                
            return True, None
            
        except Exception as e:
            return False, f"图像验证异常: {str(e)}"
    
    def _validate_csv(self, csv_path: Path) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        验证CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            (是否有效, 数据DataFrame, 错误信息)
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 检查列数 (根据dataloading.py应该有20列)
            expected_columns = 20
            if len(df.columns) != expected_columns:
                return False, None, f"CSV列数错误: {len(df.columns)} (应为{expected_columns})"
                
            # 检查是否有NaN值
            if df.isna().any().any():
                return False, None, "CSV包含NaN值"
                
            # 检查时间戳是否递增
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].values
                if not np.all(np.diff(timestamps) > 0):
                    return False, None, "时间戳不是严格递增"
                    
            return True, df, None
            
        except Exception as e:
            return False, None, f"CSV验证异常: {str(e)}"
    
    def _validate_image_csv_match(self, image_files: List[Path], csv_data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        验证图像和CSV数据的对应关系
        
        Args:
            image_files: 图像文件列表
            csv_data: CSV数据DataFrame
            
        Returns:
            (是否有效, 错误信息)
        """
        # 检查数量是否匹配
        if len(image_files) != len(csv_data):
            # 根据dataloading.py，允许最后一个图像没有对应的遥测数据
            if len(image_files) == len(csv_data) + 1:
                # 检查最后一个图像的时间戳是否大于最后一个CSV行的时间戳
                last_image_name = image_files[-1].stem
                try:
                    last_image_time = float(last_image_name)
                    last_csv_time = csv_data.iloc[-1, 1] if len(csv_data.columns) > 1 else csv_data.iloc[-1, 0]
                    
                    if last_image_time > last_csv_time:
                        return True, "最后一个图像没有对应的遥测数据（符合预期）"
                except:
                    pass
                    
            return False, f"图像数量({len(image_files)})与CSV行数({len(csv_data)})不匹配"
            
        return True, None
    
    def _generate_statistics(self):
        """生成数据集统计信息"""
        stats = {}
        
        # 基本统计
        stats["trajectory_count"] = self.results["total_trajectories"]
        stats["valid_trajectory_count"] = self.results["valid_trajectories"]
        stats["image_count"] = self.results["total_images"]
        
        # 错误统计
        stats["image_error_count"] = len(self.results["image_errors"])
        stats["csv_error_count"] = len(self.results["csv_errors"])
        stats["mismatch_error_count"] = len(self.results["mismatch_errors"])
        
        # 有效性百分比
        if stats["trajectory_count"] > 0:
            stats["valid_trajectory_percentage"] = (stats["valid_trajectory_count"] / stats["trajectory_count"]) * 100
        else:
            stats["valid_trajectory_percentage"] = 0
            
        self.results["statistics"] = stats
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("ViT-Fly 数据集验证报告")
        report.append("=" * 80)
        report.append(f"数据集目录: {self.data_dir}")
        report.append(f"验证时间: {pd.Timestamp.now()}")
        report.append("")
        
        # 统计信息
        stats = self.results["statistics"]
        report.append("统计摘要:")
        report.append(f"  轨迹总数: {stats.get('trajectory_count', 0)}")
        report.append(f"  有效轨迹: {stats.get('valid_trajectory_count', 0)} ({stats.get('valid_trajectory_percentage', 0):.1f}%)")
        report.append(f"  图像总数: {stats.get('image_count', 0)}")
        report.append("")
        
        # 错误摘要
        report.append("错误摘要:")
        report.append(f"  图像错误: {stats.get('image_error_count', 0)}")
        report.append(f"  CSV错误: {stats.get('csv_error_count', 0)}")
        report.append(f"  匹配错误: {stats.get('mismatch_error_count', 0)}")
        report.append("")
        
        # 详细错误
        if self.results["image_errors"]:
            report.append("图像错误详情:")
            for error in self.results["image_errors"][:10]:  # 只显示前10个错误
                report.append(f"  - {error}")
            if len(self.results["image_errors"]) > 10:
                report.append(f"  ... 还有 {len(self.results['image_errors']) - 10} 个错误")
            report.append("")
            
        if self.results["csv_errors"]:
            report.append("CSV错误详情:")
            for error in self.results["csv_errors"][:10]:
                report.append(f"  - {error}")
            if len(self.results["csv_errors"]) > 10:
                report.append(f"  ... 还有 {len(self.results['csv_errors']) - 10} 个错误")
            report.append("")
            
        if self.results["mismatch_errors"]:
            report.append("匹配错误详情:")
            for error in self.results["mismatch_errors"][:10]:
                report.append(f"  - {error}")
            if len(self.results["mismatch_errors"]) > 10:
                report.append(f"  ... 还有 {len(self.results['mismatch_errors']) - 10} 个错误")
            report.append("")
        
        # 建议
        report.append("建议:")
        if stats.get("valid_trajectory_percentage", 0) < 80:
            report.append("  ⚠️  数据集有效性较低，建议检查数据源")
        if stats.get("image_error_count", 0) > 0:
            report.append("  ⚠️  存在图像格式错误，建议重新生成或转换图像")
        if stats.get("csv_error_count", 0) > 0:
            report.append("  ⚠️  存在CSV格式错误，建议检查数据采集脚本")
        if stats.get("trajectory_count", 0) == 0:
            report.append("  ❌ 未找到任何轨迹数据，需要下载或生成数据集")
            report.append("     从 https://upenn.app.box.com/v/ViT-quad-datashare 下载数据 (密码: vitfly2025)")
            report.append("     解压命令: unzip data.zip -d training/datasets/data")
        elif stats.get("valid_trajectory_percentage", 0) >= 90:
            report.append("  ✅ 数据集质量良好，适合训练")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # 写入文件
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"报告已保存到: {output_file}")
            
        return report_text

def main():
    """主函数"""
    # 检查数据集目录
    data_dir = "training/datasets/data"
    
    print("ViT-Fly 数据集验证工具")
    print("=" * 50)
    
    # 创建验证器
    validator = DatasetValidator(data_dir)
    
    # 验证数据集
    print("开始验证数据集...")
    results = validator.validate_dataset()
    
    # 生成报告
    print("\n生成验证报告...")
    report = validator.generate_report("dataset_validation_report.txt")
    
    print(report)
    
    # 返回退出码
    stats = results["statistics"]
    if stats.get("valid_trajectory_count", 0) == 0:
        print("\n❌ 数据集验证失败: 没有有效轨迹")
        return 1
    elif stats.get("valid_trajectory_percentage", 0) < 50:
        print("\n⚠️  数据集验证警告: 有效性较低")
        return 2
    else:
        print("\n✅ 数据集验证通过")
        return 0

if __name__ == "__main__":
    sys.exit(main())