#!/bin/bash
cd /root/vitfly/training
exec python3 -u train_mamba_optimized.py --data_dir /root/vitfly/training/datasets/data_full --branches A B C D E --epochs 100 --batch_size 64 --lr 0.0003 --num_workers 2 --seed 42 > logs/run_nw2.log 2>&1
