#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== OpenCode 便携式配置脚本 ===${NC}"
echo ""

if [ -z "$OPENCODE" ]; then
    echo -e "${RED}错误: 未检测到 OpenCode 环境${NC}"
    echo "请在 OpenCode 终端中运行此脚本"
    exit 1
fi

CONFIG_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
ORCHESTRA_DIR="${ORCHESTRA_SKILLS_DIR:-$HOME/.orchestra}"
AGENTS_DIR="${AGENTS_SKILLS_DIR:-$HOME/.agents}"

echo -e "${BLUE}配置目录:${NC}"
echo "  OpenCode: $CONFIG_DIR"
echo "  Claude:   $CLAUDE_DIR"
echo "  Orchestra: $ORCHESTRA_DIR"
echo "  Agents:   $AGENTS_DIR"
echo ""

read -p "是否使用以上路径? (y/n) [y]: " confirm
confirm=${confirm:-y}
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "请设置环境变量后重新运行:"
    echo "  export OPENCODE_CONFIG_DIR=/your/path"
    echo "  export CLAUDE_CONFIG_DIR=/your/path"
    exit 0
fi

echo -e "${YELLOW}步骤 1/7: 创建配置目录${NC}"
mkdir -p "$CONFIG_DIR"
mkdir -p "$CLAUDE_DIR"
mkdir -p "$ORCHESTRA_DIR/skills"
mkdir -p "$AGENTS_DIR/skills"

echo -e "${YELLOW}步骤 2/7: 检测现有配置${NC}"
if [ -f "$CONFIG_DIR/opencode.json" ]; then
    echo -e "${YELLOW}警告: 发现现有配置文件${NC}"
    read -p "是否备份并覆盖? (y/n) [n]: " overwrite
    overwrite=${overwrite:-n}
    if [[ $overwrite =~ ^[Yy]$ ]]; then
        backup_dir="$HOME/opencode-backup-$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        cp "$CONFIG_DIR/opencode.json" "$backup_dir/" 2>/dev/null || true
        cp "$CONFIG_DIR/oh-my-openagent.json" "$backup_dir/" 2>/dev/null || true
        echo "已备份到: $backup_dir"
    else
        echo "跳过配置文件生成"
        exit 0
    fi
fi

echo -e "${YELLOW}步骤 3/7: 配置 opencode.json${NC}"
cat > "$CONFIG_DIR/opencode.json" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "oh-my-openagent@latest"
  ],
  "provider": {
    "anthropic": {
      "options": {
        "apiKey": "YOUR_API_KEY_HERE",
        "baseURL": "https://api.anthropic.com"
      }
    }
  },
  "permission": {
    "read": {
      "~/.config/opencode/get-shit-done/*": "allow"
    },
    "external_directory": {
      "~/.config/opencode/get-shit-done/*": "allow"
    }
  }
}
EOF

echo -e "${YELLOW}步骤 4/7: 配置 oh-my-openagent.json${NC}"
cat > "$CONFIG_DIR/oh-my-openagent.json" << 'EOF'
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-openagent/dev/assets/oh-my-opencode.schema.json",
  "agents": {
    "hephaestus": {"model": "anthropic/claude-sonnet-4-6"},
    "oracle": {"model": "anthropic/claude-sonnet-4-6"},
    "librarian": {"model": "anthropic/claude-sonnet-4-6"},
    "explore": {"model": "anthropic/claude-sonnet-4-6"},
    "multimodal-looker": {"model": "anthropic/claude-sonnet-4-6"},
    "prometheus": {"model": "anthropic/claude-sonnet-4-6"},
    "metis": {"model": "anthropic/claude-sonnet-4-6"},
    "momus": {"model": "anthropic/claude-sonnet-4-6"},
    "atlas": {"model": "anthropic/claude-sonnet-4-6"},
    "sisyphus-junior": {"model": "anthropic/claude-sonnet-4-6"}
  },
  "categories": {
    "visual-engineering": {"model": "anthropic/claude-sonnet-4-6"},
    "ultrabrain": {"model": "anthropic/claude-sonnet-4-6"},
    "deep": {"model": "anthropic/claude-sonnet-4-6"},
    "artistry": {"model": "anthropic/claude-sonnet-4-6"},
    "quick": {"model": "anthropic/claude-sonnet-4-6"},
    "unspecified-low": {"model": "anthropic/claude-sonnet-4-6"},
    "unspecified-high": {"model": "anthropic/claude-sonnet-4-6"},
    "writing": {"model": "anthropic/claude-sonnet-4-6"}
  }
}
EOF

echo -e "${YELLOW}步骤 5/7: 安装 oh-my-openagent 插件${NC}"
cd "$CONFIG_DIR"
if [ ! -d "node_modules" ]; then
    if command -v npm &> /dev/null; then
        npm install @opencode-ai/plugin@1.4.6
    else
        echo -e "${YELLOW}警告: npm 未安装,跳过插件安装${NC}"
    fi
fi

echo -e "${YELLOW}步骤 6/7: 配置 Claude 设置${NC}"
if [ ! -f "$CLAUDE_DIR/settings.json" ]; then
    cat > "$CLAUDE_DIR/settings.json" << 'EOF'
{
  "hooks": {
    "SessionStart": [],
    "PostToolUse": [],
    "PreToolUse": []
  }
}
EOF
else
    echo "Claude settings.json 已存在,跳过"
fi

echo -e "${YELLOW}步骤 7/7: 创建技能符号链接${NC}"
mkdir -p "$CONFIG_DIR/skills"
skill_count=0

for skills_dir in "$ORCHESTRA_DIR/skills" "$AGENTS_DIR/skills"; do
    if [ -d "$skills_dir" ]; then
        for skill_dir in "$skills_dir"/*; do
            if [ -d "$skill_dir" ]; then
                skill_name=$(basename "$skill_dir")
                target="$CONFIG_DIR/skills/$skill_name"
                if [ ! -e "$target" ]; then
                    ln -s "$skill_dir" "$target"
                    ((skill_count++))
                fi
            fi
        done
    fi
done

echo "创建了 $skill_count 个技能符号链接"

echo ""
echo -e "${GREEN}=== 配置完成 ===${NC}"
echo ""
echo -e "${YELLOW}下一步操作:${NC}"
echo "1. 编辑 API 密钥:"
echo "   nano $CONFIG_DIR/opencode.json"
echo ""
echo "2. 调整模型配置:"
echo "   nano $CONFIG_DIR/oh-my-openagent.json"
echo ""
echo "3. 重启 OpenCode"
echo ""
echo -e "${GREEN}配置文件位置:${NC}"
echo "  - OpenCode: $CONFIG_DIR/opencode.json"
echo "  - Agents: $CONFIG_DIR/oh-my-openagent.json"
echo "  - Claude: $CLAUDE_DIR/settings.json"
echo "  - Skills: $CONFIG_DIR/skills/ ($skill_count 个)"
