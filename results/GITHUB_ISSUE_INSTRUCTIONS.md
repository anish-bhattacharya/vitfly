# Manual GitHub Issue Creation Instructions

## Issue Details

**Repository**: https://github.com/Liber1917/vitfly  
**Title**: Branch C Weight File Incomplete - Requires Complete Retraining

## Steps to Create Issue

1. **Navigate to repository**:
   ```
   https://github.com/Liber1917/vitfly/issues/new
   ```

2. **Copy title**:
   ```
   Branch C Weight File Incomplete - Requires Complete Retraining
   ```

3. **Copy body from**:
   ```
   /root/catkin_ws/src/vitfly-mambatest/results/GITHUB_ISSUE_BRANCH_C.md
   ```

4. **Add labels**:
   - `bug`
   - `priority: high`
   - `training`
   - `branch-C`
   - `needs-retraining`

5. **Set milestone** (if available):
   - Mamba Branch Retraining

6. **Assign** (optional):
   - Leave unassigned or assign to appropriate team member

## Quick Copy-Paste

### Title
```
Branch C Weight File Incomplete - Requires Complete Retraining
```

### Body
The complete issue body is in:
```
/root/catkin_ws/src/vitfly-mambatest/results/GITHUB_ISSUE_BRANCH_C.md
```

### Labels (comma-separated)
```
bug, priority: high, training, branch-C, needs-retraining
```

## Alternative: Using GitHub CLI (if available)

If you have the official GitHub CLI (`gh`) installed:

```bash
gh issue create \
  --repo Liber1917/vitfly \
  --title "Branch C Weight File Incomplete - Requires Complete Retraining" \
  --body-file results/GITHUB_ISSUE_BRANCH_C.md \
  --label "bug" \
  --label "priority: high" \
  --label "training" \
  --label "branch-C" \
  --label "needs-retraining"
```

## Alternative: Using GitHub API with Token

If you have a GitHub personal access token:

```bash
# Set your token
export GITHUB_TOKEN="your_token_here"

# Create issue
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/Liber1917/vitfly/issues \
  -d @- << 'EOF'
{
  "title": "Branch C Weight File Incomplete - Requires Complete Retraining",
  "body": "$(cat results/GITHUB_ISSUE_BRANCH_C.md)",
  "labels": ["bug", "priority: high", "training", "branch-C", "needs-retraining"]
}
EOF
```

## Verification

After creating the issue:
1. Note the issue number (e.g., #42)
2. Verify all labels are applied
3. Confirm the body rendered correctly (especially code blocks and tables)
4. Add to project board if applicable

## Related Files

- Issue draft: `results/GITHUB_ISSUE_BRANCH_C.md`
- Weight verification: `results/WEIGHT_FILE_VERIFICATION_REPORT.md`
- Retraining plan: `results/RETRAINING_PLAN.md`
- Test report: `results/MAMBA_BRANCH_TEST_REPORT.md`
