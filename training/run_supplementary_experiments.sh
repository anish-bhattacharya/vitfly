#!/bin/bash
# Supplementary experiments for peer review revision
# ==================================================
# Run on GPU machine. Simulation tests go on WSL2 machine.
#
# Workflow:
#   1. Run P0 tasks (parallel, no dependencies)
#   2. Run P1 tasks (after P0 completes)
#   3. Push results: bash ../verify_and_push.sh
#
# P0: Multi-seed verification (statistical rigor)
# P1: G_basic+distillation (ablation)
#
# Each task is self-contained for independent execution.

set -e
cd "$(dirname "$0")"

# ─── Config ──────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/root/vitfly/training/datasets/data_full}"
SEEDS="${SEEDS:-42 43 44 45}"
EPOCHS_BC="${EPOCHS_BC:-100}"
EPOCHS_DISTILL="${EPOCHS_DISTILL:-50}"
DEVICES="${DEVICES:-0}"  # comma-separated, or "0 1" for multi-GPU

# ─── Helper ───────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
run_bc() {
    local branch=$1 seed=$2 gpu=$3
    log "P0: BC $branch seed=$seed on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train_mamba_optimized.py \
        --branches "$branch" --seed "$seed" \
        --epochs "$EPOCHS_BC" --data_dir "$DATA_DIR"
}
run_distill() {
    local branch=$1 seed=$2 gpu=$3
    log "P0: Distill $branch seed=$seed on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train_distill.py \
        --branch "$branch" --seed "$seed" \
        --epochs "$EPOCHS_DISTILL" --data_dir "$DATA_DIR" \
        --alpha 1 --beta 1 --gamma 1
}

# ─── P0: Multi-seed verification ─────────────────────────────────────────
# These are independent — run in parallel across GPUs if available.
# ======================================================================

p0_basic() {
    # G_basic BC × 5 seeds (~5h total)
    for seed in $SEEDS; do
        run_bc G "$seed" "$DEVICES"
    done
    log "P0: G_basic BC done"
}

p0_bplus() {
    # B+ BC + Distill × 5 seeds (~15h total)
    for seed in $SEEDS; do
        run_bc Bplus "$seed" "$DEVICES"
        run_distill Bplus "$seed" "$DEVICES"
    done
    log "P0: B+ BC+Distill done"
}

p0_e() {
    # E BC + Distill × 5 seeds (~10h total)
    for seed in $SEEDS; do
        run_bc E "$seed" "$DEVICES"
        run_distill E "$seed" "$DEVICES"
    done
    log "P0: E BC+Distill done"
}

# ─── P1: Supplementary ablations ─────────────────────────────────────────
# Dependencies: P0 must complete first (provides BC pretrained models)
# ======================================================================

p1_g_distill() {
    # G_basic + Distill × 1 seed (~2h)
    # Requires G BC pretrained model from P0
    local seed="${1:-42}"
    run_distill G "$seed" "$DEVICES"
    log "P1: G_basic+Distill seed=$seed done"
}

# ─── Dispatch ────────────────────────────────────────────────────────────
# By default, run P0 sequentially on one GPU.
# If 3+ GPUs available, set DEVICES="0 1 2" to parallelize branches.
# ======================================================================

case "${1:-all}" in
    all)
        p0_basic
        p0_bplus
        p0_e
        p1_g_distill
        ;;
    p0_basic)    p0_basic ;;
    p0_bplus)    p0_bplus ;;
    p0_e)        p0_e ;;
    p1_g_distill) p1_g_distill "$2" ;;
    *)
        echo "Usage: $0 {all|p0_basic|p0_bplus|p0_e|p1_g_distill [seed]}"
        exit 1
        ;;
esac

log "All tasks complete. Run verify_and_push.sh to upload results."
