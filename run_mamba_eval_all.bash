#!/bin/bash
# Evaluate all 5 Mamba branches using the proven launch_mamba_evaluation.bash pipeline.
# Usage: ./run_mamba_eval_all.bash <N_rollouts> [branches]
# Example: ./run_mamba_eval_all.bash 10
# Example: ./run_mamba_eval_all.bash 5 A B C
#
# REQUIRED before running:
#   ip addr add 192.168.233.250/32 dev lo
#   source /root/catkin_ws/devel/setup.bash

N="${1:-5}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments/mamba_branches/optimized_training"
RESULTS_DIR="$SCRIPT_DIR/results/mamba_eval_$(date '+%m%d_%H%M')"
mkdir -p "$RESULTS_DIR"

declare -A BRANCH_MODEL_TYPE=(
  [A]="VMambaLSTM"
  [B]="MambaVisionSSM"
  [C]="CNNMamba3"
  [D]="STHMamba"
  [E]="DecisionMamba"
)

if [ $# -eq 0 ]; then
  BRANCHES=(A B C D E)
else
  BRANCHES=("$@")
fi

echo "========================================"
echo " Mamba Branch Evaluation"
echo " Rollouts per branch: $N"
echo " Branches: ${BRANCHES[*]}"
echo " Results: $RESULTS_DIR"
echo "========================================"
echo ""

# Verify IP alias is set
if ! ip addr show lo | grep -q "192.168.233.250"; then
  echo "ERROR: ROS IP alias missing. Run:"
  echo "  ip addr add 192.168.233.250/32 dev lo"
  exit 1
fi

for BRANCH in "${BRANCHES[@]}"; do
  MODEL_TYPE="${BRANCH_MODEL_TYPE[$BRANCH]}"
  MODEL_PATH="$EXP_DIR/branch_${BRANCH}/best_model.pth"

  if [ ! -f "$MODEL_PATH" ]; then
    echo "SKIP branch $BRANCH — weights not found: $MODEL_PATH"
    continue
  fi

  echo ""
  echo "----------------------------------------"
  echo " Branch $BRANCH | $MODEL_TYPE"
  echo " Weights: $MODEL_PATH"
  echo "----------------------------------------"

  BRANCH_RESULTS="$RESULTS_DIR/branch_${BRANCH}"
  mkdir -p "$BRANCH_RESULTS"

  cd "$SCRIPT_DIR"
  bash launch_mamba_evaluation.bash "$N" vision "" "$MODEL_TYPE" "$MODEL_PATH" \
    2>&1 | tee "$BRANCH_RESULTS/eval.log"

  # Copy evaluation summary if present
  [ -f evaluation.yaml ] && cp evaluation.yaml "$BRANCH_RESULTS/evaluation.yaml"

  echo "Branch $BRANCH done. Results saved to $BRANCH_RESULTS"
done

echo ""
echo "========================================"
echo " All branches complete. Results in:"
echo " $RESULTS_DIR"
echo "========================================"
