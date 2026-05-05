#!/bin/bash
# Sequence Distillation: seq_len=16, BC-initialized, ViT+LSTM teacher
#
# Combines multi-step sequence pre-training with knowledge distillation.
# Uses the seq_len=16 BC checkpoint as student initialization.
#
# Usage: bash run_seq16_distill.sh
#
# Output: experiments/mamba_branches/optimized_training/seq16_distill_E/
#
# Reference (single-step distillation):
#   python3 train_distill.py --branch E --epochs 50  # sequential baseline

set -e
cd /root/vitfly/training

BRANCH=E
SEQ_LEN=16
EPOCHS=50
SAVE_DIR="/root/vitfly/experiments/mamba_branches/optimized_training/seq16_distill_E"
BC_INIT="$SAVE_DIR/seq16_bc_init.pth"
LOG="logs/seq16_distill_E_$(date +%m%d_%H%M).log"

echo "[$(date)] Starting sequence distillation: seq_len=$SEQ_LEN, Branch $BRANCH"
echo "  Teacher: ViT+LSTM"
echo "  Student init: seq_len=16 BC checkpoint"
echo "  Output: $SAVE_DIR"

OMP_NUM_THREADS=1 python3 -u train_distill.py \
  --branch $BRANCH \
  --epochs $EPOCHS \
  --batch-size 32 \
  --num-workers 0 \
  --alpha 1.0 --beta 1.0 --gamma 1.0 \
  --sequence-length $SEQ_LEN \
  --init-from-bc \
  --teacher-ckpt /root/vitfly/models/ViTLSTM_model.pth \
  --save-dir $SAVE_DIR \
  > $LOG 2>&1

echo "[$(date)] Sequence distillation complete. Results in $SAVE_DIR"
echo "  Log: $LOG"
