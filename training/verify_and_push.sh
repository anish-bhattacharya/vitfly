#!/bin/bash
LOG="/root/vitfly/training/logs/run_final.log"
CHECK="/root/vitfly/experiments/mamba_branches/optimized_training"

echo "=== 训练完成验证 ==="
for b in A B C D E; do
  pth="$CHECK/branch_$b/best_model.pth"
  if [ -f "$pth" ]; then
    sz=$(du -h "$pth" | cut -f1)
    echo "Branch $b: $sz ✓"
  else
    echo "Branch $b: ✗ MISSING"
  fi
done

echo ""
echo "=== 训练指标汇总 ==="
grep "Training Branch\|Best validation loss" "$LOG" | paste - -

echo ""
echo "=== 磁盘 ==="
df -h / | tail -1

echo ""
echo "=== Git推送 ==="
cd /root/vitfly && git add experiments/mamba_branches/optimized_training/ && git commit -m "feat: 100-epoch full training weights for all 5 branches" && git push origin mambatest && echo "✓ Push done" || echo "✗ Push failed"
