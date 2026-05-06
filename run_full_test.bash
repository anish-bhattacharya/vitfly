#!/bin/bash
# Usage: bash run_full_test.bash <BRANCH> <MODEL_TYPE> [VARIANT] [DES_VEL] [SEQ_LEN]
#   VARIANT: empty/"bc" → best_model.pth, "distill" → distill_best_model.pth
#   DES_VEL: desired velocity (default: 5.0)
#   SEQ_LEN: frames per inference (default: 1 = single-step)
# Example:
#   bash run_full_test.bash B MambaVisionSSM                  # BC @ 5m/s, seq_len=1
#   bash run_full_test.bash B MambaVisionSSM distill          # distill @ 5m/s
#   bash run_full_test.bash B MambaVisionSSM distill 7.0      # distill @ 7m/s
#   bash run_full_test.bash B MambaVisionSSM distill 7.0 8    # distill @ 7m/s, seq_len=8
BRANCH=$1
MODEL_TYPE=$2
VARIANT=${3:-""}
DES_VEL=${4:-5.0}
SEQ_LEN=${5:-1}

BASE_DIR="/root/catkin_ws/src/vitfly-mambatest/experiments/mamba_branches/optimized_training/branch_${BRANCH}"
if [ -n "$VARIANT" ] && [ "$VARIANT" != "bc" ]; then
  MODEL_PATH="${BASE_DIR}/${VARIANT}_best_model.pth"
  SUMMARY_TAG="${VARIANT}"
else
  MODEL_PATH="${BASE_DIR}/best_model.pth"
  SUMMARY_TAG="bc"
fi

export ROS_MASTER_URI=http://192.168.233.250:11311
export ROS_IP=192.168.233.250
unset ROS_HOSTNAME
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_py38
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH

# Reset drone
rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
sleep 1
rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
sleep 1
rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
sleep 1
rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once
sleep 2

cd /root/catkin_ws/src/vitfly-mambatest/envtest/ros

python3 evaluation_node.py branch_${BRANCH}_${SUMMARY_TAG} > /tmp/eval_${BRANCH}.log 2>&1 &
EVAL_PID=$!

python3 -u run_competition.py --vision_based --des_vel ${DES_VEL} \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --seq-len ${SEQ_LEN} \
  > /tmp/comp_${BRANCH}.log 2>&1 &
COMP_PID=$!

for i in $(seq 1 30); do
  sleep 2
  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once > /dev/null 2>&1
  if ! kill -0 $EVAL_PID 2>/dev/null; then
    echo "Eval finished"
    break
  fi
done

kill -SIGINT $COMP_PID 2>/dev/null
sleep 2

echo "=== Branch $BRANCH Summary ==="
cat /root/catkin_ws/src/vitfly-mambatest/envtest/ros/summary.yaml
echo "=== Velocity outputs: $(grep -c "RUN_COMPETITION.*velocity" /tmp/comp_${BRANCH}.log) ==="

# Save summary (with variant tag to avoid confusion)
cp /root/catkin_ws/src/vitfly-mambatest/envtest/ros/summary.yaml \
  /root/catkin_ws/src/vitfly-mambatest/results/branch_${BRANCH}_${SUMMARY_TAG}_summary.yaml
