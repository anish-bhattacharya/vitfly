#!/bin/bash

# DroneMamba 训练脚本
echo "======================================"
echo "Training DroneMamba for UAV Obstacle Avoidance"
echo "======================================"

cd /root/.lingma/worktree/vitfly/XBSDYR/training

# 运行训练
python3 train.py --config config/train_mamba.txt \
    --basedir /root/.lingma/worktree/vitfly/XBSDYR \
    --datadir /root/.lingma/worktree/vitfly/XBSDYR/envtest/ros/train_set \
    --logdir training/logs \
    --model_type DroneMamba \
    --lr 1e-3 \
    --N_eps 60 \
    --batch_size 32 \
    --lr_warmup_epochs 5 \
    --save_model_freq 10 \
    --val_freq 5

echo "======================================"
echo "Training complete!"
echo "Model saved in: /root/.lingma/worktree/vitfly/XBSDYR/training/logs/"
echo "======================================"
