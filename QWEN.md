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
