# Retest Report — Latest Weights (Commit 536e738)

**Test Date:** 2026-05-03
**Working Directory:** `/root/catkin_ws/src/vitfly-mambatest`
**Git Commit:** 536e738 (latest mambatest with Branch B+ and updated weights)

## Executive Summary

🎉 **MAJOR IMPROVEMENT: 5/6 branches succeed (83% success rate)**

Compared to previous 100-epoch test (only 1/5 success), the latest weights show dramatic improvement:
- **Branch B+ (NEW)**: First test, success on first try
- **Branches D & E**: Regression FIXED (was failing in 100-epoch test)
- **Branch C**: First successful pass (always crashed before)
- **Branch A**: Only remaining failure — has lateral steering bias issue

## Detailed Results

| Branch | Model | Success | Crashes | Time | vy Range | vz Range | Velocity Outputs |
|--------|-------|---------|---------|------|----------|----------|------------------|
| A | VMambaLSTM | ❌ Failed | 1 | 4.32s | 0.015 (constant) | 0.04-0.05 | 145 |
| B | MambaVisionSSM | ✅ Success | 0 | 4.22s | 0.16-0.22 | -0.26~-0.12 | 242 |
| **B+** | **BPlusModel** | ✅ **Success** | 0 | 4.24s | 0.20-0.21 | -0.20~-0.17 | 241 |
| C | CNNMamba3 | ✅ Success | 0 | 4.22s | -0.30~-0.06 | -0.29~+0.20 | 220 |
| D | STHMamba | ✅ Success | 0 | 4.21s | -0.03~+0.05 | -0.15~-0.10 | 243 |
| E | DecisionMamba | ✅ Success | 0 | 4.21s | -0.12~-0.09 | -0.05~+0.03 | 243 |

## Comparison: 100-Epoch (Old) vs Latest Weights

| Branch | 100-Epoch Result | Latest Result | Change |
|--------|------------------|---------------|--------|
| A | ❌ Failed (vy=0.02-0.04) | ❌ Failed (vy=0.015) | No improvement — pitch-only |
| B | ✅ Success | ✅ Success | Maintained |
| B+ | N/A (didn't exist) | ✅ Success | NEW — works first try |
| C | ❌ Failed | ✅ **Success** | ✅ FIXED |
| D | ❌ Regression | ✅ **Success** | ✅ REGRESSION FIXED |
| E | ❌ Regression | ✅ **Success** | ✅ REGRESSION FIXED |

**Success Rate**: 20% → **83%** 🚀

## Critical Issue: Branch A Lateral Bias

**User-confirmed observation**: "A分支似乎只会俯仰" (Branch A only seems to pitch)

**Evidence from velocity outputs**:
- vy ≈ 0.015 (essentially constant, near-zero lateral)
- vz ≈ 0.04-0.05 (small positive vertical)
- Forward velocity vx = 1.0 (clipped to max)

**Implication**: Branch A model has **lateral dimension collapse**. It learned to:
- Move forward (correct)
- Make small vertical adjustments (pitch)
- But **fails to produce lateral velocity** (vy ≈ 0)

This is different from the original mode collapse — vy is consistently near-zero rather than copying input. The model never learned to use the lateral dimension for obstacle avoidance.

## Hypothesis for Branch A Failure

Possible causes (for training pipeline review):

1. **Architectural issue**: VMambaLSTM's recurrent state may suppress lateral signal
2. **Training data imbalance**: vy values in training data may be too small/rare
3. **Loss weighting**: vy errors may contribute too little to total loss
4. **Learning rate**: Too high LR may have caused vy weights to not converge
5. **Hidden state initialization**: LSTM state may bias toward zero lateral output

## Architecture Comparison

| Branch | Stateful? | Lateral Performance | Notes |
|--------|-----------|---------------------|-------|
| A | Yes (LSTM) | ❌ Near-zero vy | LSTM may suppress lateral signal |
| B | No | ✅ Strong positive vy | Stable left steering |
| B+ | No | ✅ Strong positive vy | MambaVision+Mamba3 hybrid works |
| C | No | ✅ Wide vy range | Most dynamic responses |
| D | No | ✅ Mild vy variation | Conservative but works |
| E | No | ✅ Negative vy bias | Slight left tendency |

**Pattern**: All stateless models work; only stateful (LSTM) model fails on lateral.

## Recommendations

### Immediate
1. ✅ **Deploy Branches B, B+, C, D, E** for production testing
2. 🔍 **Branch A needs targeted investigation** — focus on:
   - Why vy stays near-zero across all depth inputs
   - Whether LSTM hidden state is biasing the output
   - Training loss decomposition (vx vs vy vs vz contributions)

### Training Pipeline Review

For Branch A specifically:
1. **Check vy loss curve** — does vy loss converge or stall early?
2. **Inspect LSTM hidden state** — is it always near zero?
3. **Try removing LSTM** — test if a stateless variant works
4. **Per-dimension loss weighting** — `loss = w_x*Lx + w_y*Ly + w_z*Lz` with `w_y > 1`
5. **Data analysis** — what fraction of training samples have |vy| > 0.1?

## Files

- Test results: `results/branch_*_retest_summary.yaml`
- This report: `results/RETEST_REPORT.md`
- Test logs: `/tmp/branch_*_epoch1.log`

## Test Configuration

- Simulator: Flightmare on WSL2 (Ubuntu Noetic)
- Network: Fixed IP 192.168.233.250
- Goal distance: 20m
- Desired velocity: 5.0 m/s
- Test script: `test_mamba_branch.bash`
- Skill used: `.claude/skills/vitfly/SKILL.md`

---

**Generated:** 2026-05-03
**Tested by:** vitfly skill automated test suite
**Weights commit:** 536e738
