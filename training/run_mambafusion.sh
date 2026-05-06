#!/bin/bash
set -e
SAVE="/root/vitfly/experiments/mamba_branches/optimized_training"
LOG="/root/vitfly/training/logs"
CONFIG='mambavision_config.in_channels=1 mambavision_config.stem_dim=48 mambavision_config.stage_dims=64,128,192 mambavision_config.depths=2,2,2 mambavision_config.d_state=12'

# Phase 1: MambaFusion BC ×3 seeds
echo "[$(date)] Phase 1: MambaFusion BC (seed 42 43 44)..."
for SEED in 42 43 44; do
  OMP_NUM_THREADS=1 setsid python3 -u /root/vitfly/training/train_mamba_optimized.py \
    --data_dir /root/vitfly/training/datasets/data_full \
    --branches Fusion \
    --epochs 100 --batch_size 64 --lr 0.0001 --num_workers 2 \
    --clip_grad_norm 0.5 --seed $SEED \
    --save_dir ${SAVE}/mambafusion_bc_s${SEED} \
    > ${LOG}/mambafusion_bc_s${SEED}_$(date +%m%d_%H%M).log 2>&1 &
  sleep 5
done
echo "BC ×3 launched. Waiting..."
wait
echo "[$(date)] Phase 1 done."

# Phase 2: MambaFusion Distill ×3 seeds
echo "[$(date)] Phase 2: MambaFusion Distill (seed 42 43 44)..."
for SEED in 42 43 44; do
  mkdir -p ${SAVE}/mambafusion_distill_s${SEED}/branch_Fusion
  OMP_NUM_THREADS=1 setsid python3 -u /root/vitfly/training/train_distill.py \
    --branch Fusion \
    --epochs 50 --batch-size 32 --num-workers 2 \
    --alpha 1.0 --beta 1.0 --gamma 1.0 \
    --seed $SEED \
    --teacher-ckpt /root/vitfly/models/ViTLSTM_model.pth \
    --save-dir ${SAVE}/mambafusion_distill_s${SEED} \
    > ${LOG}/mambafusion_distill_s${SEED}_$(date +%m%d_%H%M).log 2>&1 &
  sleep 5
done
echo "Distill ×3 launched. Waiting..."
wait
echo "[$(date)] Phase 2 done. All MambaFusion experiments complete."
