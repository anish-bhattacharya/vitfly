#!/bin/bash

# Set ROS environment for WSL
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME

# WSL2 mirrored mode: fix 127.0.0.1 routing (loops through loopback0 instead of lo,
# which breaks NetMQ Signaler's internal TCP loopback and prevents Unity-ZMQ connection)
if ip route get 127.0.0.1 2>/dev/null | grep -q loopback0; then
  ip route del 127.0.0.1 via 169.254.73.152 dev loopback0 proto kernel src 127.0.0.1 onlink table 127 2>/dev/null
  ip route flush cache 2>/dev/null
  echo "[LAUNCH SCRIPT] Fixed 127.0.0.1 routing (was loopback0, now lo)"
fi

# WSL2 graphics: force Mesa OpenGL over NVIDIA GLX (which crashes with XWayland)
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Force absolute path for WSL
export FLIGHTMARE_PATH=/root/catkin_ws/src/vitfly/flightmare

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=5
fi

echo $2

if [ "$2" = "vision" ]
then
  echo
  echo "[LAUNCH SCRIPT] Vision based!"
  echo
  run_competition_args="--vision_based"
  realtimefactor=""
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  run_competition_args="--state_based"
  if [ "$3" = "human" ]
  then
    run_competition_args="--keyboard"
    realtimefactor="real_time_factor:=1.0"
  else
    run_competition_args=""
    realtimefactor="real_time_factor:=10.0"
  fi
else
  echo
  echo "[LAUNCH SCRIPT] Unknown or empty second argument: $2, only 'vision' or 'state' allowed!"
  echo
  exit 1
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
  if [ -z $(pgrep visionsim_node) ]
  then
    roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
    ROS_PID="$!"
    echo $ROS_PID
    sleep 15
  else
    ROS_PID=""
  fi

SUMMARY_FILE="evaluation.yaml"
echo "" > $SUMMARY_FILE

# generate datetime string to label summary folders with in evaluation_node.py
datetime=$(date '+d%m_%d_t%H_%M')

relaunch_sim=0

for i in $(eval echo {1..$N})
do
  # Reset the simulator if needed
  if ((relaunch_sim))
  then
      echo
      echo
      echo
      echo
      echo RELAUNCHING SIMULATOR ON RUN $i
      echo
      echo
      echo
      echo

    # reset flag and kill everything to restart
    relaunch_sim=0
    killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
    sleep 10

    # Launch the simulator, unless it is already running
    if [ -z $(pgrep visionsim_node) ]
    then
  roslaunch envsim visionenv_sim.launch render:=False gui:=False rviz:=True $realtimefactor &
      ROS_PID="$!"
      echo $ROS_PID
      sleep 10
    else
      killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
      sleep 10
    fi

  fi

  start_time=$(date +%s)

  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
  # Add conda packages AFTER ROS packages in PYTHONPATH  
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
  python3 evaluation_node.py ${datetime}_N$i &
  PY_PID="$!"

  export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7  
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
  python3 run_competition.py $run_competition_args --des_vel 5.0 --model_type "${4:-ViTLSTM}" --model_path "${5:-../../models/ViTLSTM_model.pth}" &
  COMP_PID="$!"

  cd -

  sleep 2

  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    echo
    echo [LAUNCH_EVALUATION] Sending start navigation command
    echo
    rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
    sleep 2

    # if the current iteration has surpassed the time limit, something went wrong (possibly: [Pipeline]     Bridge failed!). Kill the simulator.
    if ((($(date +%s) - start_time) >= 300))
    then
      echo
      echo
      echo
      echo
      echo "Time limit exceeded. Exiting evaluation script loop."
      echo
      echo
      echo
      echo
      kill -SIGINT $PY_PID
      relaunch_sim=1
      break
    fi

  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi

