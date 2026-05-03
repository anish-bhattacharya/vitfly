# OpenCode 配置脚本环境路径分析

## 📊 环境特定路径使用情况

### ✅ **良好实践 - 使用环境变量**

两个脚本都正确使用了 `$HOME` 环境变量,使得脚本可以在不同用户环境下工作:

```bash
CONFIG_DIR="$HOME/.config/opencode"
CLAUDE_DIR="$HOME/.claude"
ORCHESTRA_DIR="$HOME/.orchestra"
AGENTS_DIR="$HOME/.agents"
```

### ⚠️ **潜在问题 - 假设特定目录结构**

#### 1. **GSD 权限路径** (opencode.json 行 52, 55)
```json
"~/.config/opencode/get-shit-done/*": "allow"
```
**问题**: 假设 GSD 安装在这个特定位置
**影响**: 如果用户的 GSD 安装在其他位置,权限配置会失效
**建议**: 添加路径检测或让用户自定义

#### 2. **技能目录依赖** (setup-opencode.sh 行 145-154)
```bash
if [ -d "$ORCHESTRA_DIR/skills" ]; then
    for skill_dir in "$ORCHESTRA_DIR/skills"/*; do
```
**问题**: 假设 orchestra skills 存在于 `~/.orchestra/skills`
**影响**: 如果用户使用不同的技能管理方式,符号链接创建会失败
**建议**: 检测多个可能的技能位置

#### 3. **Claude 配置目录** (行 25)
```bash
CLAUDE_DIR="$HOME/.claude"
```
**问题**: 假设 Claude 配置在 `~/.claude`
**影响**: 某些安装可能使用不同的配置目录
**建议**: 添加环境变量支持

#### 4. **npm 依赖** (行 129)
```bash
npm install @opencode-ai/plugin@1.4.6
```
**问题**: 假设 npm 已安装且可用
**影响**: 如果 npm 不可用,插件安装会失败
**建议**: 添加 npm 可用性检查

## 🔧 改进方案

### 已创建的便携式脚本 (`setup-opencode-portable.sh`)

新脚本包含以下改进:

1. **环境变量支持**
   ```bash
   CONFIG_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
   CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
   ORCHESTRA_DIR="${ORCHESTRA_SKILLS_DIR:-$HOME/.orchestra}"
   AGENTS_DIR="${AGENTS_SKILLS_DIR:-$HOME/.agents}"
   ```

2. **交互式确认**
   - 显示将要使用的路径
   - 允许用户确认或取消
   - 提示如何自定义路径

3. **现有配置检测**
   - 检测是否已有配置文件
   - 提供备份选项
   - 避免意外覆盖

4. **依赖检查**
   - 检查 npm 是否可用
   - 如果不可用,跳过插件安装并警告

5. **多技能目录支持**
   - 同时检查 orchestra 和 agents 目录
   - 统计创建的符号链接数量

## 📝 使用建议

### 对于标准环境
使用原始脚本 `setup-opencode.sh`:
```bash
./setup-opencode.sh
```

### 对于自定义环境
使用便携式脚本 `setup-opencode-portable.sh`:
```bash
# 使用默认路径
./setup-opencode-portable.sh

# 或自定义路径
export OPENCODE_CONFIG_DIR=/custom/path/opencode
export CLAUDE_CONFIG_DIR=/custom/path/claude
./setup-opencode-portable.sh
```

## 🎯 关键发现总结

| 路径类型 | 硬编码程度 | 可移植性 | 建议 |
|---------|-----------|---------|------|
| 用户主目录 | ✅ 使用 $HOME | 高 | 保持现状 |
| 配置目录 | ⚠️ 固定路径 | 中 | 添加环境变量 |
| GSD 路径 | ❌ 硬编码 | 低 | 需要检测或配置 |
| 技能目录 | ⚠️ 单一位置 | 中 | 支持多位置 |
| npm 依赖 | ❌ 假设存在 | 低 | 添加检查 |

## 🚀 推荐的部署流程

1. **首次部署**: 使用 `setup-opencode-portable.sh`
2. **标准环境**: 使用 `setup-opencode.sh` (更快)
3. **自定义环境**: 设置环境变量后使用便携式脚本
4. **备份**: 始终先运行 `backup-opencode-config.sh`

## 📦 文件清单

- `setup-opencode.sh` - 原始配置脚本 (快速,标准环境)
- `setup-opencode-portable.sh` - 便携式配置脚本 (灵活,自定义环境)
- `backup-opencode-config.sh` - 配置备份工具
- `README-opencode-setup.md` - 详细使用文档
- `OPENCODE-CONFIG-ANALYSIS.md` - 本分析文档
