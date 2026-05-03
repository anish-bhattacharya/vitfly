#!/bin/bash
# 全量训练监控脚本 - 实时跟踪5个分支100 epoch训练进度

LOG_FILE="/root/vitfly/training/logs/full_training_100epochs.log"
CHECKPOINT_DIR="/root/vitfly/experiments/mamba_branches/optimized_training"

echo "=========================================="
echo "Vitfly Mamba 全量训练监控"
echo "=========================================="
echo "训练配置: 5个分支 (A, B, C, D, E) × 100 epochs"
echo "日志文件: $LOG_FILE"
echo "检查点目录: $CHECKPOINT_DIR"
echo "=========================================="
echo ""

# 检查训练进程是否运行
check_training_process() {
    if ps aux | grep -q "[p]ython3.*train_mamba_optimized.py"; then
        echo "✓ 训练进程运行中"
        return 0
    else
        echo "✗ 训练进程未运行"
        return 1
    fi
}

# 显示当前训练进度
show_progress() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "等待训练日志生成..."
        return
    fi
    
    echo "=== 最新训练进度 ==="
    tail -50 "$LOG_FILE" | grep -E "(Training Branch|Epoch.*Train Loss|Best validation loss|GPU)" | tail -20
    echo ""
}

# 显示各分支训练状态
show_branch_status() {
    echo "=== 各分支训练状态 ==="
    for branch in A B C D E; do
        if [ -d "$CHECKPOINT_DIR/branch_$branch" ]; then
            best_model="$CHECKPOINT_DIR/branch_$branch/best_model.pth"
            if [ -f "$best_model" ]; then
                size=$(du -h "$best_model" | cut -f1)
                mtime=$(stat -c %y "$best_model" | cut -d'.' -f1)
                echo "✓ Branch $branch: 已保存检查点 ($size, 更新时间: $mtime)"
            else
                echo "○ Branch $branch: 训练中..."
            fi
        else
            echo "○ Branch $branch: 等待开始..."
        fi
    done
    echo ""
}

# 显示GPU使用情况
show_gpu_status() {
    echo "=== GPU 状态 ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s: %s | 利用率: %s%% | 内存: %s/%s MB | 温度: %s°C\n", $1, $2, $3, $4, $5, $6}'
    else
        echo "nvidia-smi 不可用"
    fi
    echo ""
}

# 估算剩余时间
estimate_remaining_time() {
    if [ ! -f "$LOG_FILE" ]; then
        return
    fi
    
    # 提取已完成的epoch数和当前分支
    current_branch=$(tail -100 "$LOG_FILE" | grep "Training Branch" | tail -1 | grep -oP "Branch \K[A-E]")
    completed_epochs=$(tail -100 "$LOG_FILE" | grep -oP "Epoch\s+\K\d+(?=/100)" | tail -1)
    
    if [ -n "$current_branch" ] && [ -n "$completed_epochs" ]; then
        echo "=== 训练进度估算 ==="
        echo "当前分支: Branch $current_branch"
        echo "已完成轮数: $completed_epochs/100"
        
        # 计算分支进度
        case $current_branch in
            A) branch_num=1 ;;
            B) branch_num=2 ;;
            C) branch_num=3 ;;
            D) branch_num=4 ;;
            E) branch_num=5 ;;
        esac
        
        total_progress=$(( (branch_num - 1) * 100 + completed_epochs ))
        total_epochs=500
        progress_percent=$(( total_progress * 100 / total_epochs ))
        
        echo "总体进度: $total_progress/$total_epochs epochs ($progress_percent%)"
        echo ""
    fi
}

# 主循环
while true; do
    clear
    echo "=========================================="
    echo "Vitfly Mamba 全量训练监控"
    echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    if check_training_process; then
        show_gpu_status
        estimate_remaining_time
        show_branch_status
        show_progress
        
        echo "=========================================="
        echo "按 Ctrl+C 退出监控 (训练继续运行)"
        echo "下次更新: 30秒后"
        echo "=========================================="
    else
        echo "训练已完成或进程已停止"
        show_branch_status
        echo ""
        echo "查看完整日志: tail -f $LOG_FILE"
        break
    fi
    
    sleep 30
done
