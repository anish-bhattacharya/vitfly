#!/usr/bin/env python3
"""
ViT-Fly 数据集分析脚本
分析数据集是否适合行为克隆训练
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class DatasetAnalyzer:
    def __init__(self, data_dir: str = "training/datasets/data"):
        self.data_dir = Path(data_dir)
        self.analysis_results = {
            "trajectory_stats": [],
            "image_stats": [],
            "velocity_stats": {},
            "collision_stats": {},
            "data_quality": {}
        }
        
    def analyze_dataset(self) -> Dict:
        """分析整个数据集"""
        print(f"分析数据集: {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"错误: 数据集目录不存在")
            return self.analysis_results
            
        trajectory_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        
        all_velocities = []
        all_collisions = []
        trajectory_lengths = []
        
        for traj_folder in trajectory_folders:
            print(f"分析轨迹: {traj_folder.name}")
            
            # 分析单个轨迹
            traj_stats = self._analyze_trajectory(traj_folder)
            self.analysis_results["trajectory_stats"].append(traj_stats)
            
            # 收集统计数据
            if "velocities" in traj_stats:
                all_velocities.extend(traj_stats["velocities"])
            if "collision_count" in traj_stats:
                all_collisions.append(traj_stats["collision_count"])
            if "length" in traj_stats:
                trajectory_lengths.append(traj_stats["length"])
        
        # 计算总体统计
        if all_velocities:
            velocities = np.array(all_velocities)
            self.analysis_results["velocity_stats"] = {
                "mean": float(np.mean(velocities)),
                "std": float(np.std(velocities)),
                "min": float(np.min(velocities)),
                "max": float(np.max(velocities)),
                "median": float(np.median(velocities))
            }
        
        if all_collisions:
            collisions = np.array(all_collisions)
            self.analysis_results["collision_stats"] = {
                "total_collisions": int(np.sum(collisions)),
                "collision_trajectories": int(np.sum(collisions > 0)),
                "collision_rate": float(np.mean(collisions > 0))
            }
        
        # 评估数据质量
        self._evaluate_data_quality(trajectory_lengths)
        
        return self.analysis_results
    
    def _analyze_trajectory(self, traj_folder: Path) -> Dict:
        """分析单个轨迹"""
        stats = {
            "name": traj_folder.name,
            "length": 0,
            "collision_count": 0,
            "velocities": [],
            "image_mean": 0,
            "image_std": 0
        }
        
        # 检查CSV文件
        csv_file = traj_folder / "data.csv"
        if not csv_file.exists():
            return stats
            
        try:
            df = pd.read_csv(csv_file)
            stats["length"] = len(df)
            
            # 分析速度数据
            if "desired_vel_x" in df.columns:
                velocities = np.sqrt(df["desired_vel_x"]**2 + df["desired_vel_y"]**2 + df["desired_vel_z"]**2)
                stats["velocities"] = velocities.tolist()
            
            # 分析碰撞数据
            if "collision" in df.columns:
                stats["collision_count"] = int(df["collision"].sum())
            
            # 分析图像数据
            image_files = sorted(traj_folder.glob("*.png"))
            if image_files:
                image_values = []
                for img_file in image_files[:10]:  # 只分析前10张图像以减少计算量
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        image_values.extend(img.flatten() / 255.0)
                
                if image_values:
                    stats["image_mean"] = float(np.mean(image_values))
                    stats["image_std"] = float(np.std(image_values))
                    
        except Exception as e:
            print(f"  分析轨迹 {traj_folder.name} 时出错: {e}")
            
        return stats
    
    def _evaluate_data_quality(self, trajectory_lengths: List[int]):
        """评估数据质量"""
        if not trajectory_lengths:
            self.analysis_results["data_quality"] = {
                "score": 0,
                "issues": ["无轨迹数据"],
                "recommendations": ["下载或生成数据集"]
            }
            return
            
        lengths = np.array(trajectory_lengths)
        
        issues = []
        recommendations = []
        score = 100  # 初始分数
        
        # 检查轨迹长度
        avg_length = np.mean(lengths)
        min_length = np.min(lengths)
        
        if avg_length < 30:
            issues.append(f"平均轨迹长度较短 ({avg_length:.1f}帧)")
            recommendations.append("收集更长的轨迹数据")
            score -= 20
            
        if min_length < 10:
            issues.append(f"存在过短轨迹 ({min_length}帧)")
            recommendations.append("移除过短轨迹")
            score -= 15
        
        # 检查轨迹数量
        num_trajectories = len(lengths)
        if num_trajectories < 10:
            issues.append(f"轨迹数量较少 ({num_trajectories}个)")
            recommendations.append("收集更多轨迹数据")
            score -= 25
            
        # 检查数据多样性
        length_std = np.std(lengths)
        if length_std / avg_length < 0.1:
            issues.append("轨迹长度过于一致")
            recommendations.append("收集不同长度的轨迹")
            score -= 10
        
        # 最终评估
        if score >= 80:
            quality = "优秀"
        elif score >= 60:
            quality = "良好"
        elif score >= 40:
            quality = "一般"
        else:
            quality = "较差"
        
        self.analysis_results["data_quality"] = {
            "score": score,
            "quality": quality,
            "total_trajectories": num_trajectories,
            "avg_trajectory_length": float(avg_length),
            "min_trajectory_length": int(min_length),
            "max_trajectory_length": int(np.max(lengths)),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def generate_analysis_report(self, output_file: Optional[str] = None) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append("ViT-Fly 数据集分析报告 (行为克隆训练适用性)")
        report.append("=" * 80)
        
        # 数据质量评估
        quality = self.analysis_results.get("data_quality", {})
        report.append("\n数据质量评估:")
        report.append(f"  综合评分: {quality.get('score', 0)}/100 ({quality.get('quality', '未知')})")
        report.append(f"  轨迹总数: {quality.get('total_trajectories', 0)}")
        report.append(f"  平均轨迹长度: {quality.get('avg_trajectory_length', 0):.1f} 帧")
        report.append(f"  最短轨迹: {quality.get('min_trajectory_length', 0)} 帧")
        report.append(f"  最长轨迹: {quality.get('max_trajectory_length', 0)} 帧")
        
        # 速度统计
        velocity_stats = self.analysis_results.get("velocity_stats", {})
        if velocity_stats:
            report.append("\n速度统计:")
            report.append(f"  平均速度: {velocity_stats.get('mean', 0):.3f} m/s")
            report.append(f"  速度标准差: {velocity_stats.get('std', 0):.3f} m/s")
            report.append(f"  速度范围: [{velocity_stats.get('min', 0):.3f}, {velocity_stats.get('max', 0):.3f}] m/s")
        
        # 碰撞统计
        collision_stats = self.analysis_results.get("collision_stats", {})
        if collision_stats:
            report.append("\n碰撞统计:")
            report.append(f"  总碰撞次数: {collision_stats.get('total_collisions', 0)}")
            report.append(f"  有碰撞的轨迹: {collision_stats.get('collision_trajectories', 0)}")
            report.append(f"  碰撞轨迹比例: {collision_stats.get('collision_rate', 0)*100:.1f}%")
        
        # 问题和建议
        if quality.get("issues"):
            report.append("\n发现的问题:")
            for issue in quality["issues"]:
                report.append(f"  ⚠️  {issue}")
        
        if quality.get("recommendations"):
            report.append("\n改进建议:")
            for rec in quality["recommendations"]:
                report.append(f"  ✅  {rec}")
        
        # 行为克隆训练适用性评估
        report.append("\n行为克隆训练适用性评估:")
        
        score = quality.get("score", 0)
        if score >= 80:
            report.append("  🎯 非常适合行为克隆训练")
            report.append("     - 数据量充足")
            report.append("     - 轨迹长度合适")
            report.append("     - 数据多样性良好")
        elif score >= 60:
            report.append("  👍 适合行为克隆训练，但有改进空间")
            report.append("     - 基本数据要求满足")
            report.append("     - 建议按照上述建议改进")
        elif score >= 40:
            report.append("  ⚠️  勉强适合行为克隆训练")
            report.append("     - 数据量或质量不足")
            report.append("     - 训练效果可能不理想")
            report.append("     - 强烈建议改进数据")
        else:
            report.append("  ❌ 不适合行为克隆训练")
            report.append("     - 数据严重不足")
            report.append("     - 需要重新收集数据")
        
        # ViT-Fly特定要求
        report.append("\nViT-Fly项目特定要求检查:")
        
        # 检查图像尺寸
        report.append("  1. 图像尺寸: 60×90 ✓" if self._check_image_size() else "  1. 图像尺寸: 不符合要求 ✗")
        
        # 检查数据格式
        report.append("  2. 数据格式: PNG + CSV ✓" if self._check_data_format() else "  2. 数据格式: 不符合要求 ✗")
        
        # 检查时间戳对齐
        report.append("  3. 时间戳对齐: 需要进一步验证")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"分析报告已保存到: {output_file}")
            
        return report_text
    
    def _check_image_size(self) -> bool:
        """检查图像尺寸"""
        try:
            image_files = list(self.data_dir.rglob("*.png"))
            if not image_files:
                return False
                
            # 检查前几个图像
            for img_file in image_files[:3]:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape != (60, 90):
                    return False
            return True
        except:
            return False
    
    def _check_data_format(self) -> bool:
        """检查数据格式"""
        try:
            trajectory_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
            if not trajectory_folders:
                return False
                
            for traj_folder in trajectory_folders[:3]:
                csv_file = traj_folder / "data.csv"
                png_files = list(traj_folder.glob("*.png"))
                
                if not csv_file.exists() or not png_files:
                    return False
                    
            return True
        except:
            return False

def main():
    """主函数"""
    print("ViT-Fly 数据集分析工具")
    print("=" * 50)
    
    analyzer = DatasetAnalyzer("training/datasets/data")
    
    print("开始分析数据集...")
    results = analyzer.analyze_dataset()
    
    print("\n生成分析报告...")
    report = analyzer.generate_analysis_report("dataset_analysis_report.txt")
    
    print(report)
    
    # 检查是否适合训练
    quality = results.get("data_quality", {})
    score = quality.get("score", 0)
    
    if score >= 60:
        print("\n✅ 数据集适合行为克隆训练")
        return 0
    else:
        print("\n⚠️  数据集需要改进才能用于行为克隆训练")
        return 1

if __name__ == "__main__":
    import sys
    from typing import Optional
    sys.exit(main())