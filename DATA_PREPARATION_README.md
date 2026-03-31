# ViT-Fly Mamba分支B-E 数据准备指南

## 概述

本文档提供了ViT-Fly项目Mamba分支B-E的数据准备、验证和分析流程。包含完整的工具链用于确保数据质量适合行为克隆训练。

## 文件结构

```
vitfly/
├── check_dataset.py          # 数据验证脚本
├── analyze_dataset.py        # 数据分析脚本
├── create_sample_data.py     # 示例数据生成脚本
├── data_preparation.py       # 数据准备主脚本
├── dataset_validation_report.txt     # 验证报告
├── dataset_analysis_report.txt       # 分析报告
└── training/datasets/reports/
    └── training_preparation_report.txt  # 训练准备报告
```

## 快速开始

### 1. 验证现有数据集

```bash
# 验证数据集格式和完整性
python check_dataset.py

# 分析数据集是否适合行为克隆训练
python analyze_dataset.py

# 运行完整的数据准备流程
python data_preparation.py --action all
```

### 2. 创建示例数据集（用于测试）

```bash
# 创建示例数据集
python create_sample_data.py

# 或使用数据准备脚本
python data_preparation.py --create-sample --action validate
```

### 3. 下载真实数据集

根据项目README，从以下位置下载真实数据集：
- **URL**: https://upenn.app.box.com/v/ViT-quad-datashare
- **密码**: vitfly2025
- **文件**: `data.zip` (2.5GB)

解压命令：
```bash
mkdir -p training/datasets/data
unzip data.zip -d training/datasets/data
```

## 数据格式要求

### 目录结构
```
training/datasets/data/
├── 0001/                    # 轨迹文件夹（数字命名）
│   ├── 0.000000.png        # 深度图像（时间戳命名）
│   ├── 0.100000.png
│   └── data.csv            # 遥测数据
├── 0002/
└── ...
```

### 图像要求
- **格式**: PNG
- **尺寸**: 60×90 像素
- **类型**: 灰度深度图像
- **值范围**: 0-255

### CSV格式要求
- **列数**: 20列
- **列头**: `timestamp,ros_time,desired_vel_x,desired_vel_y,desired_vel_z,q_w,q_x,q_y,q_z,curr_vel_x,curr_vel_y,curr_vel_z,curr_pos_x,curr_pos_y,curr_pos_z,ct_br_x,ct_br_y,ct_br_z,ct_br_w,collision`
- **数据**: 无NaN值，时间戳递增

## 验证内容

### 1. 基本验证 (`check_dataset.py`)
- 目录结构检查
- 图像格式验证（尺寸、类型、值范围）
- CSV格式验证（列数、数据类型、NaN值）
- 图像-CSV对应关系验证

### 2. 深度分析 (`analyze_dataset.py`)
- 轨迹统计（数量、长度分布）
- 速度统计分析
- 碰撞统计分析
- 数据质量评估
- 行为克隆训练适用性评估

### 3. ViT-Fly特定检查
- 图像尺寸: 60×90 ✓
- 数据格式: PNG + CSV ✓
- 时间戳对齐: 需要验证

## 数据质量评估标准

### 评分系统 (0-100分)
- **≥80分**: 优秀 - 非常适合训练
- **60-79分**: 良好 - 适合训练，有改进空间
- **40-59分**: 一般 - 勉强适合训练
- **<40分**: 较差 - 不适合训练

### 评估因素
1. **轨迹数量**: ≥10个轨迹为佳
2. **轨迹长度**: 平均≥30帧为佳
3. **数据多样性**: 轨迹长度应有变化
4. **数据完整性**: 无缺失文件或损坏数据
5. **格式正确性**: 符合ViT-Fly项目要求

## 训练准备

### 生成的配置文件
运行数据准备脚本后，会生成：
- `training/config/train_mamba_branch.txt` - Mamba分支训练配置模板

### 训练建议
根据数据集规模：

| 数据规模 | 图像数量 | 批次大小 | 训练轮数 | 用途 |
|---------|---------|---------|---------|------|
| 小型 | <1000 | 8-16 | 20-50 | 测试/调试 |
| 中型 | 1000-10000 | 16-32 | 50-100 | 训练/验证 |
| 大型 | >10000 | 32-64 | 100-200 | 完整训练 |

### Mamba分支特定建议
1. **模型选择**: ViTLSTM 或 DroneMamba
2. **输入处理**: 图像除以255.0归一化
3. **输出**: 线性速度命令 (v_x, v_y, v_z)
4. **数据增强**: 可考虑旋转、平移、亮度调整

## 常见问题

### Q1: 数据集验证失败怎么办？
**A**: 检查以下问题：
1. 图像尺寸是否为60×90
2. CSV文件是否有20列
3. 图像和CSV文件数量是否匹配
4. 是否有NaN值或损坏文件

### Q2: 数据质量评分低怎么办？
**A**: 
1. 收集更多轨迹数据
2. 确保轨迹长度有变化
3. 检查数据采集过程
4. 移除损坏或异常的轨迹

### Q3: 如何生成自己的数据集？
**A**: 
1. 使用Flightmare仿真器收集数据
```bash
bash launch_evaluation.bash 10 state
```
2. 数据会自动保存到 `envtest/ros/train_set/`
3. 移动到数据集目录：
```bash
mv envtest/ros/train_set/* training/datasets/new_dataset/
```

### Q4: 训练时遇到数据加载错误？
**A**: 
1. 确保数据集路径正确
2. 检查`training/config/train.txt`中的`dataset`参数
3. 运行验证脚本检查数据格式
4. 查看dataloading.py中的具体错误信息

## 下一步

1. **验证数据**: 运行 `python check_dataset.py`
2. **分析质量**: 运行 `python analyze_dataset.py`
3. **准备训练**: 运行 `python data_preparation.py --action prepare`
4. **开始训练**: `python training/train.py --config training/config/train_mamba_branch.txt`
5. **监控训练**: `tensorboard --logdir training/logs`
6. **测试模型**: `bash launch_evaluation.bash 1 vision`

## 参考

- [ViT-Fly项目主页](https://www.anishbhattacharya.com/research/vitfly)
- [论文](https://arxiv.org/abs/2405.10391)
- [Flightmare仿真器](https://github.com/uzh-rpg/flightmare)
- [原始数据集](https://upenn.app.box.com/v/ViT-quad-datashare)

---

**最后更新**: 2026-03-31  
**版本**: 1.0  
**维护者**: ViT-Fly开发团队