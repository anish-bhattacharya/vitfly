# ViT-Fly 调试总结

> **警告**：本文档记录了 ViT-Fly 仓库在当前环境中遇到的关键问题及解决方案。请在修改任何配置前仔细阅读，避免重复踩坑。

## 问题概述

原版 ViT-Fly 仓库设计为在完整配置的 catkin 工作空间中运行，包含 Unity 渲染器。在当前环境中，由于路径配置、环境变量和渲染设置的问题，导致评估流程无法正常运行。

---

## 关键问题及解决方案

### 1. FLIGHTMARE_PATH 路径错误

**问题描述**：
- 脚本中 FLIGHTMARE_PATH 指向 `/home/vitfly/flightmare`
- 但实际环境配置文件位于 `~/catkin_ws/src/vitfly/flightmare`
- 导致模拟器无法找到环境配置 (`environment_50/dynamic_obstacles.yaml`)

**症状**：
```
[VisionEnv] Configuring dynamic objects from: /root/catkin_ws/src/vitfly/flightmare/flightpy/configs/vision/spheres_medium/environment_50/dynamic_obstacles.yaml
```

**解决方案**：
修改 `launch_evaluation.bash`，强制使用正确的绝对路径：

```bash
# Set Flightmare Path - must point to complete flightmare with Unity resources
export FLIGHTMARE_PATH=$HOME/catkin_ws/src/vitfly/flightmare
```

---

### 2. Unity 渲染默认禁用

**问题描述**：
- 脚本默认使用 `render:=False`
- 导致无法看到深度图像和 Unity 渲染窗口
- 模拟器输出显示 `Unity Render: 0`

**症状**：
```
[VisionEnv]    [loadParam] Unity Render: 0
```

**解决方案**：
修改 `launch_evaluation.bash`，添加 RENDER 环境变量控制：

```bash
# Enable Unity rendering by default (set RENDER=0 to disable)
ENABLE_RENDER=${RENDER:-1}

# Launch the simulator
if [ "$ENABLE_RENDER" = "1" ]; then
    roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
else
    roslaunch envsim visionenv_sim.launch render:=False gui:=False rviz:=True $realtimefactor &
fi
```

**使用方法**：
```bash
# 启用渲染（默认）
export RENDER=1
bash launch_evaluation.bash 1 vision

# 禁用渲染（更稳定，但无可视化）
export RENDER=0
bash launch_evaluation.bash 1 vision
```

---

### 3. 模型类型硬编码

**问题描述**：
- `launch_evaluation.bash` 硬编码使用 ViTLSTM 模型
- 无法直接运行 DroneMamba 或其他模型

**解决方案**：
修改 `launch_evaluation.bash`，添加环境变量支持：

```bash
# Default to ViTLSTM, allow override via MODEL_TYPE environment variable
MODEL_TYPE=${MODEL_TYPE:-"ViTLSTM"}
MODEL_PATH=${MODEL_PATH:-"../../models/ViTLSTM_model.pth"}

python3 run_competition.py $run_competition_args --des_vel 7.0 --model_type "$MODEL_TYPE" --model_path "$MODEL_PATH" &
```

**使用方法**：
```bash
# 运行 ViTLSTM（默认）
bash launch_evaluation.bash 1 vision

# 运行 DroneMamba
export MODEL_TYPE="DroneMamba"
export MODEL_PATH="../../models/drone_mamba_latest.pth"
bash launch_evaluation.bash 1 vision
```

---

### 4. 进程冲突导致 Unity 崩溃

**问题描述**：
- 如果之前运行的 ROS 进程未正确关闭
- 会导致 Unity 渲染器连接失败或崩溃

**解决方案**：
在启动评估前杀死所有相关进程：

```bash
killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare. visionsim_node 2>/dev/null
sleep 3
```

---

## 正确的启动流程

### 步骤 1：清理进程
```bash
pkill -9 -f roscore
pkill -9 -f rosmaster
pkill -9 -f rosout
pkill -9 -f visionsim
pkill -9 -f rviz
sleep 2
```

### 步骤 2：设置环境变量
```bash
cd /home/vitfly
export FLIGHTMARE_PATH=$HOME/catkin_ws/src/vitfly/flightmare
export RENDER=1  # 启用 Unity 渲染
```

### 步骤 3：运行评估
```bash
# ViTLSTM（默认）
bash launch_evaluation.bash 1 vision

# 或 DroneMamba
export MODEL_TYPE="DroneMamba"
export MODEL_PATH="../../models/drone_mamba_latest.pth"
bash launch_evaluation.bash 1 vision
```

---

## 关键日志解读

### 正常启动日志
```
[VisionEnv]    [loadParam] Unity Render: 1          # 渲染已启用
[UnityBridge]  Flightmare Unity is connected.       # Unity 连接成功
[RUN_COMPETITION] Model loaded                      # 模型加载成功
[RUN_COMPETITION] Published velocity: [x, y, z]     # 速度命令发布
[Pilot]        Not in hover, won't switch to velocity reference!  # 警告，可忽略
```

### 评估结果解读
```yaml
rollout_1:
  Success: false           # 是否成功到达目标
  number_crashes: 6        # 碰撞次数
  time_to_finish: 9.27     # 完成时间（秒）
```

---

## 已知限制

1. **渲染模式不稳定**：在某些环境下，Unity 渲染可能导致崩溃。建议先在 `RENDER=0` 模式下测试。

2. **飞控状态警告**：`Not in hover, won't switch to velocity reference!` 是正常警告，只要模型继续发布速度命令即可。

3. **评估超时**：默认超时时间为 40 秒（见 `evaluation_config.yaml`）。

---

## 重要文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 启动脚本 | `/home/vitfly/launch_evaluation.bash` | 主评估启动脚本 |
| 评估配置 | `/home/vitfly/envtest/ros/evaluation_config.yaml` | 评估参数配置 |
| 模型目录 | `/home/vitfly/models/` | 预训练模型权重 |
| Unity 资源 | `~/catkin_ws/src/vitfly/flightmare/flightrender/` | Unity 渲染器二进制 |

---

## 调试命令

```bash
# 查看 ROS 话题
rostopic list

# 查看模拟器日志
ls -la /root/.ros/log/latest/

# 检查进程
ps aux | grep -E "(vision|ros)"

# 手动启动模拟器（前台）
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True
```

---

## 总结

本次调试的核心问题是：
1. **路径配置**：必须使用 catkin_ws 中的完整路径
2. **渲染设置**：需要显式启用 Unity 渲染
3. **环境变量**：通过环境变量控制模型类型和渲染开关

遵循上述流程，应该能够顺利运行 ViT-Fly 的评估流程。