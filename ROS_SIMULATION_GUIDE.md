# DroneMamba ROS 仿真测试指南

## 前提条件

1. **ROS Noetic 已安装并配置**
```bash
source /opt/ros/noetic/setup.bash
```

2. **Catkin 工作空间已编译**
```bash
cd ~/catkin_ws
catkin build
source devel/setup.bash
```

3. **DroneMamba 模型已训练**
```bash
# 模型权重路径
checkpoints/drone_mamba_latest.pth
```

## 快速开始

### 方法 1: 一键运行脚本（推荐）

```bash
cd /home/vitfly
source devel/setup.bash

./envtest/ros/run_mamba_simulation.sh \
    --model checkpoints/drone_mamba_latest.pth \
    --episodes 50 \
    --env spheres
```

### 方法 2: 分步执行

#### 步骤 1: 启动 ROS
```bash
roscore &
```

#### 步骤 2: 启动仿真环境
```bash
cd /home/vitfly
source devel/setup.bash

# 启动仿真（spheres 或 trees 环境）
roslaunch envsim envsim.launch world_name:=spheres
```

#### 步骤 3: 运行评估
```bash
python3 envtest/ros/run_mamba_competition.py \
    --model_path checkpoints/drone_mamba_latest.pth \
    --num_episodes 50 \
    --output_dir results/mamba_eval
```

## 输出结果

评估完成后，结果保存在 `results/mamba_eval_YYYYMMDD_HHMMSS/` 目录：

```
results/mamba_eval_20260317_150000/
├── statistics.json          # 统计数据
├── evaluation_report.txt    # 评估报告
├── episode_001.csv          # 第 1 次飞行数据
├── episode_002.csv          # 第 2 次飞行数据
└── ...
```

### 统计数据示例

```json
{
  "total_episodes": 50,
  "success_rate": 0.88,
  "collision_rate": 0.12,
  "avg_flight_time": 12.5,
  "avg_distance": 65.3
}
```

## 故障排除

### 问题 1: ROS 节点未就绪
```bash
# 检查 roscore 是否运行
pgrep -x roscore

# 如果没有运行，启动它
roscore &
```

### 问题 2: 仿真节点未启动
```bash
# 检查 ROS 话题
rostopic list

# 应该看到 /kingfisher/state 和 /kingfisher/depth_image
```

### 问题 3: 模型加载失败
```bash
# 检查模型文件是否存在
ls -la checkpoints/drone_mamba_latest.pth

# 如果不存在，先训练模型
cd training
python3 train.py --config config/train_mamba.txt
```

## 仿真环境选项

| 环境名称 | 描述 | 难度 |
|---------|------|------|
| `spheres` | 球体障碍物 | 简单 |
| `trees` | 树木障碍物 | 中等 |

## 评估指标

- **成功率**: 到达目标且未碰撞的飞行次数 / 总次数
- **碰撞率**: 发生碰撞的次数 / 总次数
- **平均飞行时间**: 成功飞行的平均持续时间（秒）
- **平均飞行距离**: 成功飞行的平均路径长度（米）

## 相关脚本

| 脚本 | 功能 |
|------|------|
| `run_mamba_simulation.sh` | 一键运行完整仿真流程 |
| `run_mamba_competition.py` | ROS 评估节点 |
| `test_mamba_model.py` | 简化测试（不依赖 Gazebo） |
| `quick_simulation_test.py` | 快速验证（无 ROS） |

## 参考资料

- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - 完整实验指南
- [FINAL_SIMULATION_REPORT.md](FINAL_SIMULATION_REPORT.md) - 仿真报告
- [MAMBA_IMPLEMENTATION_SUMMARY.md](MAMBA_IMPLEMENTATION_SUMMARY.md) - 实施总结
