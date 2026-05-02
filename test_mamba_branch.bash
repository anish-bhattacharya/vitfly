#!/bin/bash
# Usage: bash test_mamba_branch.bash <BRANCH> <MODEL_TYPE>
# Example: bash test_mamba_branch.bash C CNNMamba3

BRANCH=$1
MODEL_TYPE=$2
MODEL_PATH="/root/catkin_ws/src/vitfly-mambatest/experiments/mamba_branches/optimized_training/branch_${BRANCH}/best_model.pth"
LOG="/tmp/branch_${BRANCH}_epoch1.log"

# Fixed IP (matches loopback alias)
export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
export FLIGHTMARE_PATH=/root/catkin_ws/src/vitfly/flightmare
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450

ip addr add 192.168.233.250/32 dev lo 2>/dev/null

# Fix 127.0.0.1 routing
if ip route get 127.0.0.1 2>/dev/null | grep -q loopback0; then
  ip route del 127.0.0.1 via 169.254.73.152 dev loopback0 proto kernel src 127.0.0.1 onlink table 127 2>/dev/null
  ip route flush cache 2>/dev/null
fi

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH

echo "=== Testing Branch $BRANCH ($MODEL_TYPE) ===" | tee $LOG

# Kill old processes
killall -9 roscore rosmaster rosout visionsim_node rviz flight_render 2>/dev/null
pkill -9 -f "roslaunch|evaluation_node|run_competition" 2>/dev/null
sleep 5

# Launch simulator
roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True >> $LOG 2>&1 &
SIM_PID=$!
sleep 18

# Reset simulator
rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once >> $LOG 2>&1
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once >> $LOG 2>&1
rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once >> $LOG 2>&1
rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once >> $LOG 2>&1
sleep 2

cd /root/catkin_ws/src/vitfly-mambatest/envtest/ros

# Start evaluation node
python3 evaluation_node.py branch_${BRANCH}_epoch1 >> $LOG 2>&1 &
EVAL_PID=$!

# Start competition node
python3 -u run_competition.py --vision_based --des_vel 5.0 --model_type $MODEL_TYPE --model_path $MODEL_PATH >> $LOG 2>&1 &
COMP_PID=$!

# Send start navigation repeatedly
for i in $(seq 1 30); do
  sleep 2
  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once >> $LOG 2>&1
  if ! kill -0 $EVAL_PID 2>/dev/null; then
    echo "Evaluation finished." | tee -a $LOG
    break
  fi
done

kill -SIGINT $COMP_PID 2>/dev/null
kill -SIGINT $EVAL_PID 2>/dev/null
kill -SIGINT $SIM_PID 2>/dev/null
sleep 3

# Save results
cp /root/catkin_ws/src/vitfly-mambatest/envtest/ros/summary.yaml /root/catkin_ws/src/vitfly-mambatest/results/branch_${BRANCH}_epoch1_summary.yaml 2>/dev/null

echo "=== Branch $BRANCH Results ===" | tee -a $LOG
cat /root/catkin_ws/src/vitfly-mambatest/results/branch_${BRANCH}_epoch1_summary.yaml 2>/dev/null | tee -a $LOG
echo "Velocity outputs:" | tee -a $LOG
grep "RUN_COMPETITION.*velocity" $LOG | wc -l | xargs echo "  Count:" | tee -a $LOG
grep "RUN_COMPETITION.*velocity" $LOG | head -5 | tee -a $LOG
