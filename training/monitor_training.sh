#!/bin/bash
LOG_FILE="/root/vitfly/training/logs/full_retrain.log"

echo "=== Training Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

while true; do
    clear
    echo "=== Training Status ($(date '+%H:%M:%S')) ==="
    echo ""
    
    if ps aux | grep -q "[p]ython3.*train_mamba_optimized"; then
        echo "✓ Training process is RUNNING"
        PID=$(ps aux | grep "[p]ython3.*train_mamba_optimized" | awk '{print $2}')
        echo "  PID: $PID"
        ps aux | grep "[p]ython3.*train_mamba_optimized" | awk '{printf "  CPU: %s%% | MEM: %s%%\n", $3, $4}'
    else
        echo "✗ Training process NOT running"
    fi
    
    echo ""
    echo "=== Latest Log (last 15 lines) ==="
    tail -15 "$LOG_FILE" 2>/dev/null || echo "No log file yet"
    
    echo ""
    echo "Press Ctrl+C to exit monitor"
    sleep 5
done
