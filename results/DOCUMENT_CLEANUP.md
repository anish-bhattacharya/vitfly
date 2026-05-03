# Document Cleanup Record

Cleaned up outdated documents on 2026-05-03.
All deleted content reflects issues from BEFORE the SSM fix and retraining.

## Deleted Files

| File | Last Updated | Reason |
|------|-------------|--------|
| `results/comparison/` (8 files) | commit 0527d0e | Duplicate of `results/` - old upstream comparison data |
| `results/GITHUB_ISSUE_BRANCH_C.md` | commit 3fa6b95 | Branch C issues - solved (now 0 crashes in simulation) |
| `results/MAMBA_BRANCH_TEST_REPORT.md` | commit 8718ac2 | Old test report from before SSM architecture fixes |
| `results/RETRAINING_PLAN.md` | commit 8718ac2 | Retraining plan - already executed for all branches |
| `results/TRAINING_COLLAPSE_ANALYSIS.md` | commit 1eaeb11 | Mode collapse analysis - caused by broken SSM implementation, now fixed |
| `results/WEIGHT_FILE_VERIFICATION_REPORT.md` | commit d8acab5 | Old buggy weight verification - all weights retrained with correct target |
| `results/EPOCH100_TEST_REPORT.md` | commit fcf301b | Outdated test report - superseded by FULL_TEST_REPORT.md |
| `experiments/mamba_branches/BRANCH_A_ANALYSIS.md` | commit 9cc987b | Old analysis before d_state=16→64 upgrade |

## Remaining Documents (Current)

| File | Status |
|------|--------|
| `results/EXPERIMENT_REPORT.md` | ✅ Comprehensive experiment report (updated with seq_len ablation) |
| `results/FULL_TEST_REPORT.md` | ✅ Latest simulation results (5/6 branches pass) |
| `results/FUTURE_DIRECTIONS.md` | ✅ Future research directions with constraint analysis |
| `results/MULTISTEP_EXPERIMENT.md` | ✅ Multi-step sequence prediction experiment design |
| `results/RETEST_REPORT.md` | ✅ Retest report after architecture fixes |
