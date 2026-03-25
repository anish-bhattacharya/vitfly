# ViT-Fly Project Context

## Project Overview

**ViT-Fly** is the official implementation of the ICRA 2025 paper *"Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance"* from the GRASP Lab at the University of Pennsylvania.

**Purpose:** Demonstrates that Vision Transformers (ViTs) combined with LSTM recurrence layers can be trained via behavior cloning to enable end-to-end perception-based obstacle avoidance for quadrotors equipped with depth cameras. The trained policies predict linear velocity commands from depth images.

**Key Results:**
- ViT+LSTM outperforms baseline architectures (ConvNet, LSTMnet, UNet, ViT-only)
- Zero-shot transfer to real-world multi-obstacle environments
- Achieves obstacle dodging at speeds up to 7m/s

## Repository Structure

```
vitfly/
├── training/           # Model training scripts and dataloading
│   ├── train.py        # Main training script
│   ├── dataloading.py  # Dataset loading utilities
│   ├── config/         # Training configurations
│   └── logs/           # TensorBoard logs (generated)
├── models/             # Neural network architectures
│   ├── model.py        # Model definitions (ViT, ViTLSTM, etc.)
│   └── ViTsubmodules.py # ViT-specific components
├── flightmare/         # Quadrotor simulator (external submodule)
│   ├── flightlib/      # C++ physics engine
│   ├── flightrender/   # Unity rendering engine
│   └── flightros/      # ROS integration
├── envsim/             # Environment simulation launch files
├── envtest/            # Evaluation/testing in simulation
│   └── ros/            # ROS nodes for evaluation
├── depthfly/           # Real-world ROS deployment package
├── dodgedrone_simulation/  # DodgeDrone competition code
├── catkin_simple/      # Catkin build helpers
├── mav_comm/           # MAV communication protocols
├── labutils/           # Utility functions
├── analysis/           # Analysis scripts
├── models/             # Pretrained model weights (downloaded)
└── media/              # Demo GIFs and assets
```

## Building and Running

### Prerequisites

- **OS:** Ubuntu 20.04
- **ROS:** ROS Noetic
- **Python:** 3.8+
- **GPU:** Required for training (CUDA)

### Installation

1. **Set up catkin workspace:**
```bash
cd ~
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
```

2. **Clone and setup:**
```bash
cd ~/catkin_ws/src
git clone git@github.com:anish-bhattacharya/vitfly.git
cd vitfly
bash setup_ros.bash
cd ../..
catkin build
source devel/setup.bash
cd src/vitfly
```

3. **Download required assets** (from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare), password: `vitfly2025`):
```bash
# Environments (1GB)
tar -xvf <path/to/environments.tar> -C flightmare/flightpy/configs/vision

# Unity resources (450MB)
tar -xvf <path/to/flightrender.tar> -C flightmare/flightrender

# Pretrained models (50MB)
tar -xvf <path/to/pretrained_models.tar> -C models

# Training dataset (2.5GB)
mkdir -p training/datasets/data training/logs
unzip <path/to/data.zip> -d training/datasets/data
```

### Testing in Simulation

1. **Configure environment** in `flightmare/flightpy/configs/vision/config.yaml`:
```yaml
level: "spheres_medium"  # or "trees"
env_folder: "environment_<0-100>"  # or 0-499 for trees
```

2. **Run evaluation:**
```bash
bash launch_evaluation.bash <num_trials> vision
# Example: bash launch_evaluation.bash 1 vision
```

### Training

1. **Train a model:**
```bash
python training/train.py --config training/config/train.txt
```

2. **Monitor with TensorBoard:**
```bash
tensorboard --logdir training/logs
```

### Real-World Deployment

1. **Configure** `depthfly/scripts/run.py`:
   - Set `DEPTHFLY_PATH`
   - Set `self.desired_velocity`
   - Set `self.model_type` and `self.model_path`
   - Update ROS topic names

2. **Run:**
```bash
roslaunch depthfly depthfly.launch
# In another terminal:
rostopic pub -r 50 /trigger std_msgs/Empty "{}"
```

## Development Conventions

### Code Style
- Python 3 with type hints where applicable
- ROS Noetic conventions for ROS nodes
- Standard ROS package structure (CMakeLists.txt, package.xml)

### Key Dependencies
- **PyTorch** (2.4.1) - Deep learning framework
- **Transformers** (4.46.3) - HuggingFace ViT models
- **OpenCV** (4.10.0) - Image processing
- **TensorBoard** - Training visualization
- **ROS Noetic** - Robot middleware

### Architecture Patterns
- **Models:** Located in `models/`, imported via `sys.path.append`
- **Training:** Config-driven via `.txt` config files using `ConfigArgParse`
- **Evaluation:** ROS node-based with topic subscriptions/publications
- **Simulation:** Unity rendering + C++ physics engine (Flightmare)

### Testing Practices
- Simulation testing via `envtest/ros/evaluation_node.py`
- Metrics: crash counting, trial statistics, success rates
- Debug image streaming on `/debug_img1` topic

## Configuration Files

| File | Purpose |
|------|---------|
| `evaluation.yaml` | Evaluation summary output |
| `training/config/train.txt` | Training hyperparameters |
| `flightmare/flightpy/configs/vision/config.yaml` | Simulator environment config |
| `envtest/ros/evaluation_config.yaml` | Evaluation parameters |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `setup_ros.bash` | Install system dependencies, setup workspace |
| `launch_evaluation.bash` | Launch simulator and run evaluation trials |
| `training/train.py` | Main training loop with checkpointing |
| `envtest/ros/run_competition.py` | Run model inference during simulation |
| `depthfly/scripts/run.py` | Real-world depth camera inference node |

## Available Model Architectures

- **ConvNet** - CNN baseline
- **LSTMnet** - RNN-based
- **UNet** - Encoder-decoder CNN
- **ViT** - Vision Transformer
- **ViTLSTM** - ViT + LSTM (best performing)

## Common Issues

1. **Catkin build errors with Eigen:**
```bash
cd flightmare/flightlib/externals/eigen
rm -rf CMakeCache.txt CMakeFiles
catkin clean && catkin build
```

2. **Environment variables:**
```bash
export FLIGHTMARE_PATH=$PWD/flightmare
# Add to .bashrc for persistence
```

## Citation

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```

## External Resources

- [Project Page](https://www.anishbhattacharya.com/research/vitfly)
- [Paper](https://arxiv.org/abs/2405.10391)
- [Flightmare Simulator](https://github.com/uzh-rpg/flightmare)
- [ICRA 2022 DodgeDrone Competition](https://github.com/uzh-rpg/agile_flight)

## Qwen Added Memories
- 启动 flightmare 模拟器前必须先杀掉所有相关进程：killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare. visionsim_node，否则会导致进程冲突和崩溃
- Always run tests after modifying model inference code or simulation-related files.
- Before starting environment setup, check for existing installations and document what's already configured.
- Commit work incrementally after completing each major setup step (ROS installation, vitfly config, simulation running)

## Mamba/DroMamba Notes

### Architecture
- **DroneMamba**: Simplified-SSM + CNN hybrid for UAV obstacle avoidance
- **Parameters**: ~452K (85% smaller than ViT)
- **Key Components**:
  - SimplifiedSSM: Diagonal SSM with bidirectional scanning
  - SimplifiedSSMBlock: SSM + MLP + gating mechanisms
  - OverlapPatchMerging: Multi-scale feature extraction

### Training Configuration (CPU)
```txt
device = cpu
model_type = DroneMamba
lr = 1e-3
N_eps = 50
```

### Known Issues
- Inference dimension mismatch (519 vs 517) in temporal_ssm path
- Requires fix in `models/model.py` forward() method

### Training Results (5 epochs)
| Epoch | Loss | Time/epoch |
|-------|------|-----------|
| 1 | 0.0305 | 57.67s |
| 5 | 0.0230 | 71.63s |

### Files
- `models/mamba_submodules.py` - SSM implementations
- `models/model.py` - DroneMamba class definition
- `training/config/train_mamba.txt` - Training config
- `envtest/ros/run_mamba_competition.py` - Evaluation script

## Qwen Added Memories
- 启动 flightmare 模拟器前必须先杀掉所有相关进程：killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare. visionsim_node，否则会导致进程冲突和崩溃
- Always run tests after modifying model inference code or simulation-related files.
- Before starting environment setup, check for existing installations and document what's already configured.
- Commit work incrementally after completing each major setup step (ROS installation, vitfly config, simulation running)
- **在拉取任何一个仓库后记得检查其环境要求和配置，若没有对应环境或配置不成功则需要优先进行相应配置**
- 不要在没有跑通代码的情况下提交 GitHub 或者写文档，文档要么出现在计划阶段，要么出现在事后总结（AAR）的时候。
- 遇到报错先 websearch 相关文档而不是盲目执行 shell 命令，所有尝试尽量有依据。
- 当前设备可使用 GPU 0 (CUDA) 进行训练加速
- 不可以采用简化评估脚本进行仿真测试，必须使用真实的 Unity/Flightmare 仿真环境进行评估。
- 简化评估脚本的结果不可信，必须删除。
- WSL 系统支持图形界面，可以直接启动 Unity/Flightmare 仿真环境。
- 仿真评估必须参考 vitfly 仓库的 README 和相关手册进行，不要创建简化版本。

## 分支 A 仿真测试说明

分支 A (VMamba+LSTM) 的仿真测试因 WSL 图形界面限制无法执行。

**已完成的验证**:
- ✅ 离线推理测试：平均延迟 3.23ms (<5ms 要求)
- ✅ 参数量验证：684,931 (<5M 要求)
- ✅ 训练收敛：Val Loss = 0.5269

**待完成**:
- ⚠️ Unity/Flightmare 完整仿真测试（需要完整 Linux 桌面环境）

**仿真测试方法** (在完整 Linux 环境):
```bash
cd /root/catkin_ws/src/vitfly
MODEL_TYPE="VMambaLSTMNet" MODEL_PATH="models/VMambaLSTM_best.pth" bash launch_evaluation.bash 20 vision
```
