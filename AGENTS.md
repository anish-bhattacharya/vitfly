# ViT-Fly Project Knowledge Base

**Generated:** 2026-03-25
**Project:** Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance
**Paper:** ICRA 2025

## OVERVIEW

ViT-Fly demonstrates that Vision Transformers (ViTs) combined with LSTM can be trained via behavior cloning for end-to-end perception-based obstacle avoidance on quadrotors with depth cameras. The trained policies predict linear velocity commands from depth images.

## STRUCTURE

```
vitfly/
├── training/           # Model training scripts
│   ├── train.py        # Main training script
│   ├── dataloading.py  # Dataset loading
│   └── config/         # Training configs (.txt)
├── models/             # Neural network architectures
│   ├── model.py        # Model definitions (ViT, ViTLSTM, etc.)
│   └── ViTsubmodules.py
├── flightmare/         # Quadrotor simulator (external submodule)
│   ├── flightlib/      # C++ physics engine
│   ├── flightrender/   # Unity rendering engine
│   └── flightros/      # ROS integration
├── envsim/             # Environment simulation launch files
├── envtest/            # Evaluation/testing in simulation
│   └── ros/            # ROS nodes for evaluation
├── depthfly/           # Real-world ROS deployment package
├── dodgedrone_simulation/  # DodgeDrone competition code
└── models/             # Pretrained model weights
```

## KEY DEPENDENCIES

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.1 | Deep learning |
| Transformers | 4.46.3 | HuggingFace ViT |
| OpenCV | 4.10.0 | Image processing |
| TensorBoard | - | Training visualization |
| ROS Noetic | - | Robot middleware |

## AVAILABLE MODELS

- **ConvNet** - CNN baseline
- **LSTMnet** - RNN-based
- **UNet** - Encoder-decoder CNN
- **ViT** - Vision Transformer
- **ViTLSTM** - ViT + LSTM (best performing)
- **DroneMamba** - Simplified-SSM + CNN hybrid (~452K params)

## BUILD & RUN

### Setup
```bash
cd ~/catkin_ws/src
git clone git@github.com:anish-bhattacharya/vitfly.git
cd vitfly
bash setup_ros.bash
catkin build
source devel/setup.bash
```

### Download Assets
```bash
# Environments, Unity resources, pretrained models, training data
# See QWEN.md for detailed download instructions (Box password: vitfly2025)
```

### Test Simulation
```bash
bash launch_evaluation.bash <num_trials> vision
```

### Train
```bash
python training/train.py --config training/config/train.txt
tensorboard --logdir training/logs
```

## CONFIG FILES

| File | Purpose |
|------|---------|
| `evaluation.yaml` | Evaluation summary output |
| `training/config/train.txt` | Training hyperparameters |
| `flightmare/flightpy/configs/vision/config.yaml` | Simulator environment |
| `envtest/ros/evaluation_config.yaml` | Evaluation parameters |

## ARCHITECTURE PATTERTERNS

- **Models:** Located in `models/`, imported via `sys.path.append`
- **Training:** Config-driven via `.txt` config files using `ConfigArgParse`
- **Evaluation:** ROS node-based with topic subscriptions/publications
- **Simulation:** Unity rendering + C++ physics engine (Flightmare)

## IMPORTANT NOTES

### Qwen Added Memories
- 启动 flightmare 模拟器前必须先杀掉所有相关进程：
  ```bash
  killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare. visionsim_node
  ```
- 修改模型推理代码后必须运行测试
- 环境配置前检查已有安装
- 提交前确保代码能运行通
- 仿真评估必须使用真实 Unity/Flightmare 环境，简化脚本不可信

### Common Issues
1. **Catkin build errors with Eigen:**
   ```bash
   cd flightmare/flightlib/externals/eigen
   rm -rf CMakeCache.txt CMakeFiles
   catkin clean && catkin build
   ```
2. **Environment variables:**
   ```bash
   export FLIGHTMARE_PATH=$PWD/flightmare
   ```

## TESTING

- Simulation testing via `envtest/ros/evaluation_node.py`
- Metrics: crash counting, trial statistics, success rates
- Debug image streaming on `/debug_img1` topic

## CITATION

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```

## EXTERNAL RESOURCES

- [Project Page](https://www.anishbhattacharya.com/research/vitfly)
- [Paper](https://arxiv.org/abs/2405.10391)
- [Flightmare Simulator](https://github.com/uzh-rpg/flightmare)
- [ICRA 2022 DodgeDrone Competition](https://github.com/uzh-rpg/agile_flight)
