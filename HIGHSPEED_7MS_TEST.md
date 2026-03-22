# DroneMamba 7m/s 高速避障测试

按照 VitFly 论文标准实验流程，使用 ROS 仿真进行 7m/s 高速避障测试。

## 前提条件

1. **ROS Noetic 已配置**
```bash
source /opt/ros/noetic/setup.bash
```

2. **Catkin 工作空间已编译**
```bash
cd ~/catkin_ws
catkin build
source devel/setup.bash
```

3. **模型已训练**
```bash
# 模型权重路径
checkpoints/drone_mamba_latest.pth
```

## 快速开始

### 方法 1: 一键运行（推荐）

```bash
cd /home/vitfly

# 运行 10 次测试，使用 spheres_medium 环境
./run_7ms_test.sh checkpoints/drone_mamba_latest.pth 10 spheres_medium
```

### 方法 2: 分步执行

#### 步骤 1: 启动 ROS
```bash
roscore &
```

#### 步骤 2: 启动仿真
```bash
cd /home/vitfly
source devel/setup.bash

# 启动仿真环境（spheres_medium 或 trees）
roslaunch envsim envsim.launch world_name:=spheres_medium
```

#### 步骤 3: 运行 7m/s 评估
```bash
python3 envtest/ros/run_mamba_7ms_competition.py \
    --model_path checkpoints/drone_mamba_latest.pth \
    --num_episodes 50 \
    --timeout 20 \
    --output_dir results/mamba_7ms_eval
```

## 配置选项

### 仿真环境

| 环境名称 | 描述 | 难度 |
|---------|------|------|
| `spheres_medium` | 中等密度球体障碍物 | ⭐⭐ |
| `spheres_dense` | 高密度球体障碍物 | ⭐⭐⭐ |
| `trees` | 树木障碍物 | ⭐⭐⭐ |

### 速度配置

修改 `envtest/ros/run_mamba_7ms_competition.py` 中的 `--velocity` 参数：

```bash
# 5m/s 测试
python3 ... --velocity 5.0

# 7m/s 测试（默认）
python3 ... --velocity 7.0

# 10m/s 极限测试
python3 ... --velocity 10.0
```

## 输出结果

评估结果保存在 `results/mamba_7ms_YYYYMMDD_HHMMSS/` 目录：

```
results/mamba_7ms_20260317_160000/
├── statistics.json          # 统计数据
├── episode_001.csv          # 第 1 次飞行数据
├── episode_002.csv          # 第 2 次飞行数据
└── ...
```

### 统计数据示例

```json
{
  "desired_velocity": 7.0,
  "total_episodes": 50,
  "successful_flights": 44,
  "collisions": 6,
  "success_rate": 0.88,
  "collision_rate": 0.12
}
```

## 评估指标

根据 VitFly 论文标准：

- **成功率 (Success Rate)**: 到达 60m 目标且未碰撞的飞行次数 / 总次数
- **碰撞率 (Collision Rate)**: 发生碰撞的次数 / 总次数
- **平均飞行时间**: 成功飞行的平均持续时间（秒）
- **平均速度**: 60m / 平均飞行时间（m/s）

## 与 VitFly 基线对比

| 模型 | 速度 | 成功率 | 碰撞率 |
|------|------|--------|--------|
| ViT | 5m/s | ~85% | ~15% |
| ViT+LSTM | 5m/s | ~87% | ~13% |
| **DroneMamba** | **7m/s** | **>80%** | **<20%** |

## 故障排除

### 问题 1: 仿真未启动
```bash
# 检查 ROS 话题
rostopic list

# 应该看到 /kingfisher/state 和 /kingfisher/depth_image
```

### 问题 2: 模型加载失败
```bash
# 检查模型文件
ls -la checkpoints/drone_mamba_latest.pth

# 如果不存在，先训练模型
cd training
python3 train.py --config config/train_mamba.txt
```

### 问题 3: 碰撞率过高
- 检查仿真环境难度
- 确认模型已充分训练（loss < 0.001）
- 尝试降低速度至 5m/s 进行测试

## 参考资料

- [VitFly Paper](https://arxiv.org/abs/2405.10391)
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- [MAMBA_IMPLEMENTATION_SUMMARY.md](MAMBA_IMPLEMENTATION_SUMMARY.md)
