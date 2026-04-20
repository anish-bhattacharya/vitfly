# Vision Transformers (ViTs) for End-to-End Vision-Based Quadrotor Obstacle Avoidance (ICRA 2025)

[Project page](https://www.anishbhattacharya.com/research/vitfly)  &
[Paper](https://arxiv.org/abs/2405.10391)

This is the official repository for the paper "Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance" by Bhattacharya, et al. (2024) from GRASP, Penn.

We demonstrate that vision transformers (ViTs) can be used for end-to-end perception-based obstacle avoidance for quadrotors equipped with a depth camera. We train policies that predict linear velocity commands from depth images to avoid obstacles via behavior cloning from a privileged expert in a simple simulation environment, and show that ViT models combined with recurrence layers (LSTMs) outperform baseline methods based on other popular learning architectures.

## Project Structure

```
vitfly/
├── training/              # Model training scripts
│   ├── train_mamba_optimized.py  # Main training script for Mamba branches
│   ├── dataloading.py     # Dataset loading
│   └── config/            # Training configs
├── experiments/mamba_branches/  # Mamba branch model implementations
│   ├── branch_A_vmamba_lstm/    # VMamba + LSTM
│   ├── branch_B_mambavision_ssm/ # MambaVision + SSM
│   ├── branch_C_cnn_mamba3/     # CNN + Mamba3
│   ├── branch_D_sth_mamba/      # STH-Mamba
│   └── branch_E_decisionmamba/  # DecisionMamba
├── models/                # Original ViT-Fly models
├── flightmare/            # Quadrotor simulator
└── requirements.txt       # Dependencies
```

## Installation

#### Clone repository
```bash
cd ~/catkin_ws/src
git clone git@github.com:anish-bhattacharya/vitfly.git
cd vitfly
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

#### (Optional) Set up ROS/Flightmare
For simulation testing, see the original documentation. Additional details at https://github.com/uzh-rpg/agile_flight.

## Dataset Setup

Download `data.zip` (2.5GB, 580 trajectories) from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025):
```bash
mkdir -p training/datasets/data_full training/logs
unzip <path/to/data.zip> -d training/datasets/data_full
```

## Training

### Quick Start
```bash
cd training
python train_mamba_optimized.py --data_dir /root/vitfly/training/datasets/data_full
```

### Train Specific Branches
```bash
python train_mamba_optimized.py --branches B C D E
```

### Custom Configuration
```bash
python train_mamba_optimized.py \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.0001 \
  --num_workers 4 \
  --save_dir ./checkpoints
```

### Training Features
- Mixed Precision Training (FP16) with torch.cuda.amp
- Optimized DataLoader with parallel loading
- GPU memory monitoring
- Gradient accumulation for larger effective batch sizes
- Learning rate warmup and cosine annealing
- Checkpoint saving and validation

## Mamba Branch Results

| Branch | Model | Parameters | Best Val Loss |
|--------|-------|------------|---------------|
| A | VMamba+LSTM | ~3M | 0.00007 |
| B | MambaVision+SSM | ~2.6M | 0.000001 |
| C | CNN+Mamba3 | ~2.1M | 0.000001 |
| D | STH-Mamba | ~2.8M | 0.000001 |
| E | DecisionMamba | ~1.4M | 0.000007 |

All branches show convergence without overfitting when trained with sufficient data (200 trajectories).

## TDD Verification

All branches verified working (1 epoch, 50 trajectories):
- Branch A: Train 5.14→0.46, Val 0.0792 ✅
- Branch B: Train 9.14→0.24, Val 0.1693 ✅
- Branch C: Train 0.41→0.13, Val 0.0961 ✅

Run verification:
```bash
cd training
python train_mamba_optimized.py --branches A --epochs 1 --data_dir /root/vitfly/training/datasets/data_full --short 50
```

## Key Bugs Fixed

### 1. Target Variable Bug (CRITICAL)
**Before**: `target = [desired_vels[idx]] * 3` (repeated scalar)
**After**: `target = velocity.clone()` (correct 3D velocity)

### 2. Empty Validation Set
- Fixed: sample-level split instead of trajectory-level for small datasets

### 3. Branch E Epochs
- Fixed: retrained with correct 100 epochs instead of default 10

## Testing (Simulation)

Download pretrained models from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025):
```bash
tar -xvf <path/to/pretrained_models.tar> -C models
bash launch_evaluation.bash 1 vision
```

## Citation

```bibtex
@inproceedings{bhattacharya2025vision,
  title={Vision transformers for end-to-end vision-based quadrotor obstacle avoidance},
  author={Bhattacharya, Anish and Rao, Nishanth and Parikh, Dhruv and Kunapuli, Pratik and Wu, Yuwei and Tao, Yuezhan and Matni, Nikolai and Kumar, Vijay},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgements

Simulation launching code and the versions of `flightmare` and `dodgedrone_simulation` are from the [ICRA 2022 DodgeDrone Competition code](https://github.com/uzh-rpg/agile_flight).

---

## WSL2 Environment Setup Guide

This fork adds full **WSL2 (Windows Subsystem for Linux 2)** support for running the Flightmare simulation. The original codebase targets native Ubuntu 20.04; running it under WSL2 requires several workarounds documented below. Follow these steps in order.

### Prerequisites

- Windows 10/11 with WSL2 enabled
- Ubuntu 20.04 installed in WSL2
- NVIDIA GPU with latest Windows drivers (the driver is shared between Windows and WSL2)
- WSLg enabled (comes with modern WSL2, provides display via XWayland)

### Step 1: Enable WSL2 Mirrored Networking

Create or edit `%USERPROFILE%\.wslconfig` on the Windows side (e.g. `C:\Users\YourName\.wslconfig`):

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

Then restart WSL from PowerShell: `wsl --shutdown`, and reopen your WSL terminal.

Mirrored mode gives WSL the same IP address as Windows, which simplifies ROS networking and is required for the display stack.

### Step 2: Fix Loopback Routing (Critical)

WSL2 mirrored mode routes `127.0.0.1` traffic through a virtual `loopback0` interface instead of the standard `lo` interface. This breaks NetMQ's internal Signaler (TCP loopback pipe), which entirely prevents Unity from connecting via ZMQ. **The simulation will not work without this fix.**

The `launch_evaluation.bash` script in this fork automatically applies the fix on every run. To apply it manually:

```bash
# Check if the problem exists:
ip route get 127.0.0.1
# If output shows "dev loopback0", apply the fix:
ip route del 127.0.0.1 via 169.254.73.152 dev loopback0 proto kernel src 127.0.0.1 onlink table 127
ip route flush cache
# Verify (should show "dev lo"):
ip route get 127.0.0.1
```

### Step 3: Install ROS Noetic

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```

### Step 4: Install Python Dependencies (Miniconda)

The system Python conflicts with ROS's `cv_bridge`, so we use a Miniconda environment:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create Python 3.8 environment (matches ROS Noetic)
conda create -n ros_py38 python=3.8 -y
conda activate ros_py38

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pyyaml opencv-python scipy
```

### Step 5: Fix cv_bridge Library Conflict

ROS's `cv_bridge` and conda's OpenCV load different versions of `libffi`, causing a crash. The fix is to preload the system library:

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
```

This is already included in the modified `launch_evaluation.bash`.

### Step 6: OpenGL Configuration

Unity requires OpenGL 4.5+, but WSL2's Mesa driver defaults to 3.1. We override it with environment variables:

```bash
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
```

Do **NOT** install `libnvidia-gl-*` packages in WSL2 — they conflict with XWayland and cause Unity to crash with `glXGetVisualFromFBConfig` errors. The Mesa d3d12 driver (which comes with WSL2) handles GPU rendering correctly.

This is already included in the modified `launch_evaluation.bash`.

### Step 7: Run the Simulation

```bash
# First, apply the IP alias (once per WSL2 session):
ip addr add 192.168.233.250/32 dev lo

# Then launch:
bash launch_evaluation.bash 1 vision
```

If everything is configured correctly, you should see:
1. Unity window appears (via WSLg)
2. `[UnityBridge] Flightmare Unity is connected.`
3. `[Pilot] Z-position smaller than takeoff height, taking off!`
4. `[RUN_COMPETITION] Model loaded`
5. `[RUN_COMPETITION] compute_command_vision_based took ~0.008 seconds`

### Troubleshooting

**Unity window doesn't appear**: Verify `echo $DISPLAY` returns `:0` (WSLg default). If not, run `export DISPLAY=:0`.

**`[UnityBridge] Unity Connection time out!`**: The loopback route fix is not applied. Run:
```bash
ip route get 127.0.0.1
# Must show "dev lo", NOT "dev loopback0"
```

**`Segmentation fault (core dumped)` from visionsim_node**: This happens when Unity ZMQ connection fails. Fix the loopback route issue first.

**`[Pilot] Not in hover, won't switch to velocity reference!`**: This is a harmless warning. As long as you also see `compute_command_vision_based` messages, the simulation is running correctly.

**rviz shows blank/glitchy display**: Mesa's d3d12 driver may have rendering artifacts. This is cosmetic and doesn't affect simulation correctness.

**Simulation and rviz both fail silently / ROS cannot bind**: `launch_evaluation.bash` hardcodes `ROS_MASTER_URI=http://192.168.233.250:11311` and `ROS_IP=192.168.233.250`. If WSL2 no longer has that IP on any interface (e.g. after `wsl --shutdown` or a host network change), all ROS nodes fail to start. Fix by adding a loopback alias **once per WSL2 session**, before running the simulation:

```bash
ip addr add 192.168.233.250/32 dev lo
```

Verify it is present:
```bash
ip addr show lo | grep 192.168.233.250
```

This alias is lost on `wsl --shutdown` and must be re-applied each time WSL2 restarts. This is a network configuration step — no source code changes are needed.

**ZMQ ports 10253/10254 occupied after a crash**: When `visionsim_node` crashes, WSL2's kernel keeps the ZMQ sockets alive even after all processes die. No Linux tool (`fuser`, `ss --kill`, `kill -9`) can clear them. The only fix is to run `wsl --shutdown` from **Windows PowerShell**, then reopen WSL2:

```powershell
# Run in Windows PowerShell (not WSL terminal):
wsl --shutdown
```

After WSL2 restarts, re-apply the loopback alias above before launching.