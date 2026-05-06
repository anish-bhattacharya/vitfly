---
name: github-push-proxy
description: "Git push to GitHub from proxy-restricted environments. Covers network proxy setup, token-based auth, SSH fallback, and troubleshooting common push failures."
origin: local
---

# GitHub Push from Proxy-Restricted Environments

## When to Use

- Git push fails with `Connection timed out` (port 443 unreachable)
- `Permission denied (publickey)` — SSH key not authorized on GitHub
- `could not read Username` — credential helper fails in non-interactive mode
- Pushing large model weights (`.pth` files) to a GitHub branch

## Environment Detection

Run these to identify the failure mode:

```bash
# 1. Check SSH access
ssh -T git@github.com
# "Permission denied (publickey)" → key not registered on GitHub

# 2. Check HTTPS reachability
curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 https://github.com
# timeout/000 → need proxy

# 3. Check credential helper
git config --global credential.helper
# "!gh auth git-credential" → using gh auth
# "(empty)" → no helper configured

# 4. Check gh auth status
gh auth status
# Should show "Logged in to github.com as <user>"
```

## Step-by-Step Push Procedure

### Step 1: Enable Network Proxy (if needed)

```bash
source /etc/network_turbo
```

This sets `http_proxy`/`https_proxy` environment variables for the current shell.

### Step 2: Disable SSL Verification (if using proxy)

```bash
git config --global http.sslVerify false
```

Some proxies terminate SSL, causing cert errors. Disable verification (safe for LAN proxy to GitHub).

### Step 3: Get GitHub Token

**Preferred method**:
```bash
gh auth token
```
If this fails (older gh versions), read directly from config:
```bash
grep -oP 'ghp_\w+' ~/.config/gh/hosts.yml
```

### Step 4: Push with Token in URL

> **Why not credential helper?** `gh auth git-credential` requires interactive terminal input. In non-interactive shell sessions (CI, automated scripts, OpenCode tools), it fails with `could not read Username`.

```bash
TOKEN=$(grep -oP 'ghp_\w+' ~/.config/gh/hosts.yml)
git remote set-url origin "https://<USER>:${TOKEN}@github.com/<ORG>/<REPO>.git"
git push origin <BRANCH>
```

**Restore clean URL after push** (critical — token in remote URL is a security risk):
```bash
git remote set-url origin https://github.com/<ORG>/<REPO>.git
```

### Step 5: Verify Push

```bash
# Check remote branch has the latest commit
git ls-remote origin <BRANCH> | head -3

# Or compare local vs remote
git rev-parse HEAD
git ls-remote origin <BRANCH> | cut -f1
```

## Troubleshooting

### Push Hangs / No Output

Large repositories (200MB+ pack) may timeout default git operations:

```bash
# Increase HTTP post buffer for large pushes
git -c http.postBuffer=524288000 push origin <BRANCH>

# Disable compression for faster throughput
git -c core.compression=0 push origin <BRANCH>

# Combine both
git -c http.postBuffer=524288000 -c core.compression=0 push origin <BRANCH>
```

### "Authentication failed" with Token

```bash
# Check token format
grep 'oauth_token' ~/.config/gh/hosts.yml
# Must be ghp_... (GitHub PAT) or gho_... (OAuth)

# Verify token works
TOKEN=$(grep -oP 'ghp_\w+' ~/.config/gh/hosts.yml)
curl -H "Authorization: token $TOKEN" https://api.github.com/user | grep login
```

### "Failed to connect" After Proxy Source

```bash
# Verify proxy is set
echo "$http_proxy" "$https_proxy"

# Test connectivity
curl -I --connect-timeout 10 https://github.com

# Try direct with longer timeout (proxy may be slow)
curl -I --connect-timeout 30 https://github.com
```

### SSH Key Not Authorized

If you need SSH (e.g., for submodules):

```bash
# Generate temp deploy key
ssh-keygen -t ed25519 -f /tmp/gh_key -N ""

# Add to GitHub account
gh ssh-key add /tmp/gh_key.pub -t "temp-push-$(date +%s)"

# Push with that key
GIT_SSH_COMMAND="ssh -i /tmp/gh_key" git push origin <BRANCH>

# Clean up
gh ssh-key list | grep temp-push | head -1 | awk '{print $1}' | xargs gh ssh-key delete
rm -f /tmp/gh_key /tmp/gh_key.pub
```

## Reference: Auth Methods Comparison

| Method | Works Non-Interactive | Requires Setup | Token Exposure |
|--------|----------------------|---------------|----------------|
| SSH key | ✅ | Add public key to GitHub | None |
| HTTPS + credential helper | ❌ (needs TTY) | `gh auth login` | None |
| HTTPS + token in URL | ✅ | None (uses gh token) | Visible in shell history |
| HTTPS + `.netrc` | ✅ | Write token to file | File permissions |

## Security Notes

- **Always restore clean remote URL** after using token-in-URL method
- Token in `.config/gh/hosts.yml` is already stored with restricted permissions (600)
- `.pth` weight files are tracked in git by default (not gitignored) — no special LFS needed unless >100MB
- Clean shell history: `history -d $(history | tail -1 | awk '{print $1}')`

## Quick Reference

```bash
# === Full push flow (proxy + token) ===
source /etc/network_turbo
git config --global http.sslVerify false
TOKEN=$(grep -oP 'ghp_\w+' ~/.config/gh/hosts.yml)
git remote set-url origin "https://USER:${TOKEN}@github.com/ORG/REPO.git"
git push origin BRANCH
git remote set-url origin https://github.com/ORG/REPO.git

# === Restore original config ===
git config --global http.sslVerify true
```
