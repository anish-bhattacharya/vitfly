# Vision Transformers (ViTs) for End-to-End Vision-Based Quadrotor Obstacle Avoidance (ICRA 2025)

[Project page](https://www.anishbhattacharya.com/research/vitfly)  &nbsp;
[Paper](https://arxiv.org/abs/2405.10391)

This is the official repository for the paper "Vision Transformers for End-to-End Vision-Based Quadrotor Obstacle Avoidance" by Bhattacharya, et al. (2024) from GRASP, Penn. Please note that you may find plenty of legacy and messy code in this research project's codebase.

We demonstrate that vision transformers (ViTs) can be used for end-to-end perception-based obstacle avoidance for quadrotors equipped with a depth camera. We train policies that predict linear velocity commands from depth images to avoid obstacles via behavior cloning from a privileged expert in a simple simulation environment, and show that ViT models combined with recurrence layers (LSTMs) outperform baseline methods based on other popular learning architectures. Deployed on a real quadrotor, our method achieves zero-shot dodging behavior at speeds reaching 7m/s and on multi-obstacle environments.

<!-- GIFs -->

#### Generalization to simulation environments 
<img src="media/sim-trees-vitlstm.gif" width="660" height="200"> <img src="media/sim-window-vitlstm.gif" width="340" height="200">

#### Zero-shot transfer to multi-obstacle and high-speed, real-world dodging (GIFs not sped up)

<img src="media/multi-3d-3rdview.gif" width="300" height="200"> <img src="media/7ms-vitlstm.gif" width="380" height="200">

<img src="media/multi-3d-onboard.gif" width="300" height="200"> <img src="media/7ms-onboard.gif" width="380" height="200">

## Installation

Note that if you'd only like to train models, and *not* test in simulation, you can skip straight to section Train.

For Flightmare simulator installation, additional details can be found on [this page](https://github.com/uzh-rpg/agile_flight), though we also include a `requirements.txt` for your reference.

#### (optional) Set up a catkin workspace

If you'd like to start a new catkin workspace, then a typical workflow is (note that this code has only been tested with ROS Noetic and Ubuntu 20.04):
```
cd
mkdir -p catkin_ws/src
cd catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color
```

#### Clone this repository and set up

Once inside your desired workspace, clone this repository (note, we renamed it to `vitfly`):
```
cd ~/catkin_ws/src
git clone git@github.com:anish-bhattacharya/vitfly.git
cd vitfly
```

In order to replicate Unity environments similar to those we use for training and testing, you will need to download `environments.tar` (1GB) from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025) and extract it to the right location (below). We provide the medium-level spheres scene and a trees scene. Other obstacle environments are provided by [ICRA 2022 DodgeDrone Competition](https://github.com/uzh-rpg/agile_flight).
```
tar -xvf <path/to/environments.tar> -C flightmare/flightpy/configs/vision
```

You will also need to download our Unity resources and binaries. Download `flightrender.tar` (450MB) from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025)  and then:
```
tar -xvf <path/to/flightrender.tar> -C flightmare/flightrender
```

Then, install the dependencies via a given script:
```
bash setup_ros.bash
cd ../..
catkin build
source devel/setup.bash
cd src/vitfly
```

## Test (simulation)

#### Download pretrained weights

Download `pretrained_models.tar` (50MB) from [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025). This tarball includes pretrained models for ConvNet, LSTMnet, UNet, ViT, and ViT+LSTM (our best model).
```
tar -xvf <path/to/pretrained_models.tar> -C models
```

#### Edit the config file

For testing on a medium-level spheres or trees environment, edit the file `flightmare/flightpy/configs/vision/config.yaml` line 2-3 as following:
```
level: "spheres_medium" # spheres
env_folder: "environment_<any int between 0-100>"
```
```
level: "trees" # trees
env_folder: "environment_<any int between 0-499>"
```

When running the simulation (the following section), you can set any number `N` of trials to run. To run trials on the same, specified environment index, set `datagen: 1` and `rollout: 0`. To run sequentially different environment indices upon each trial, set `datagen: 0` and `rollout: 1`. For the latter more, the  environments prefixed with `custom_` are used. These set the static obstacles as dynamic, so that they can be moved to new positions upon each trial, within the same Unity instance.

You may also change the `unity: scene: 2` scene index according to those provided in the Unity binary. The available environments and their drone starting positions are found in `flightmare/flightpy/configs/scene.yaml`.

#### Run the simulation

The `launch_evaluation.bash` script launches Flightmare and the trained model for depth-based flight when using `vision` mode. To run one trial, run:
```
bash launch_evaluation.bash 1 vision
```

Some details: Change `1` to any number of trials you'd like to run. If you look at the bash script, you'll see multiple python scripts being run. `envtest/ros/evaluation_node.py` counts crashes, starts and aborts trials, and prints other statistics to the console. `envtest/ros/run_competition.py` subscribes to input depth images and passes them to the corresponding functions (located in `envtest/ros/user_code.py`) that run the model and return desired velocity commands. The topic `/debug_img1` streams a depth image with an overlaid velocity vector arrow which indicates the model's output velocity command.

## Train

#### Download and set up our dataset

The training dataset is available as `data.zip` (2.5GB, 3.4GB unzipped) from the [Datashare](https://upenn.app.box.com/v/ViT-quad-datashare) (pw: vitfly2025). Make some necesary directories and unzip this data (this may take some time):
```
mkdir -p training/datasets/data training/logs
unzip <path/to/data.zip> -d training/datasets/data
```

This dataset contains 580 trajectories in various sphere environments. Each numerically-named folder within `data` contains an expert trajectory of images and telemetry data. `<timestamp>.png` files are depth images and `data.csv` stores velocity commands, orientations, and other telemetry information.

#### Train a model

We provide a script `train.py` that trains a model on a given dataset, with arguments parsed from a config file. We provide `training/config/train.txt` with some default hyperparameters. Note that we have only confirmed functionality with a GPU. To train:
```
python training/train.py --config training/config/train.txt
```

You can monitor training and validation statistics with Tensorboard:
```
tensorboard --logdir training/logs
```

#### Gather your own dataset in simulation

To create your own dataset, launch the simulation in state mode (after making any desired edits to the chosen environment, camera parameters, or environment switching behavior in the flightmare config file as described in the Test section) to run our simple, privileged expert policy. Note that this included look-ahead expert policy has a limited horizon and may occasionally crash.
```
bash launch_evaluation.bash 10 state
```
The saved depth images and telemetry get automatically stored in `envtest/ros/train_set` in a format readily usable for training. Move relevant trajectory folders to your new dataset directory. If you have previously cleared the `train_set` directory, you can do `mv envtest/ros/train_set/* training/datasets/new_dataset/`. Then, simple edit your config file `dataset = new_dataset` and run the training command as in the previous section.

## Real-world deployment

We provide a simple ROS1 package for running the trained models on an incoming depth camera stream. This package, called `depthfly`, can be easily modified for your use-case. On a 12-core, 16GB RAM, cpu-only machine (similar to that used for hardware experiments) the most complex model ViT+LSTM should take around 25ms for a single inference.

You should modify the `DEPTHFLY_PATH`, `self.desired_velocity`, `self.model_type`, and `self.model_path` in the ROS node python script `depthfly/scripts/run.py`. Additionally, you need to modify the ROS topic names in the subscribers and publishers as appropriate.

Run a Realsense D435 camera and the model inference node with:
```
roslaunch depthfly depthfly.launch
```

We include a `/trigger` signal that, when continuously published to, will route the predicted velocity commands to a given topic `/robot/cmd_vel`. We do this with the following terminal command typically sent form a basestation computer. If you Ctl+C this command, the rosnode will send velocity 0.0 commands to stop in place.
```
rostopic pub -r 50 /trigger std_msgs/Empty "{}"
```

Some details:
- Keep in mind the model is trained to continuously fly at the desired velocity and would require manual pilot takeover to stop.
- Raw outputs of the node are published on `/output` topics.
- For ease-of-use, z-velocity commands are currently set to maintain a constant flight altitude of 1m (line 159, `run.py`) but can be re-written to accept the model prediction `self.pred_vel[2]`.
- We use a ramp-up in the `run()` function to smoothly accelerate the drone to the desired velocity over 2 seconds.
- Velocity commands are published with respect to x-direction forward, y-direction left, and z-direction up.

Please fly safely!

## Citation

If you found this code useful in your work then please consider citing the following paper:

```
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

### Some debugging tips below

#### `catkin build` error on existing `eigen` when building flightlib
Error message:
```
CMake Error: The current CMakeCache.txt directory vitfly/flightmare/flightlib/externals/eigen/CMakeCache.txt is different than the directory <some-other-package>/eigen where CMakeCache.txt was created. This may result in binaries being created in the wrong place. If you are not sure, reedit the CMakeCache.txt.
```
Possible solution:
```
cd flightmare/flightlib/externals/eigen
rm -rf CMakeCache.txt CMakeFiles
cd <your-workspace>
catkin clean
catkin build
```

#### `[Pilot]        Not in hover, won't switch to velocity reference!` (warning)
You can ignore this warning as long as further console prints appear indicating the sending of start navigation command, and the running of the compute_command_vision_based model.

#### `[readTrainingObs] Configuration file � does not exists.` (warning)
This appears when you are in `datagen: 1, rollout: 0` mode, and the scene manager looks for a `custom_` prefixed scene to load which is needed for `datagen: 0, rollout: 1` mode. You can ignore this warning.
