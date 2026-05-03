#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== OpenCode 完整配置脚本 (零依赖版本) ===${NC}"
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

echo -e "${YELLOW}步骤 1/9: 创建配置目录${NC}"
mkdir -p "$CONFIG_DIR"
mkdir -p "$CLAUDE_DIR"
mkdir -p "$ORCHESTRA_DIR/skills"
mkdir -p "$AGENTS_DIR/skills"
mkdir -p "$CONFIG_DIR/skills"

echo -e "${YELLOW}步骤 2/9: 检测现有配置${NC}"
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

echo -e "${YELLOW}步骤 3/9: 配置 opencode.json${NC}"
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
      "~/.config/opencode/*": "allow",
      "~/.claude/*": "allow",
      "~/.orchestra/*": "allow",
      "~/.agents/*": "allow"
    },
    "external_directory": {
      "~/.config/opencode/*": "allow",
      "~/.claude/*": "allow"
    }
  }
}
EOF

echo -e "${YELLOW}步骤 4/9: 配置 oh-my-openagent.json${NC}"
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

echo -e "${YELLOW}步骤 5/9: 检查并安装 npm${NC}"
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}npm 未安装,尝试安装...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y nodejs npm
    elif command -v yum &> /dev/null; then
        sudo yum install -y nodejs npm
    elif command -v brew &> /dev/null; then
        brew install node
    else
        echo -e "${RED}无法自动安装 npm,请手动安装 Node.js${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}步骤 6/9: 安装 oh-my-openagent 插件${NC}"
cd "$CONFIG_DIR"
if [ ! -d "node_modules/@opencode-ai/plugin" ]; then
    npm install @opencode-ai/plugin@1.4.6
    echo "✓ 插件安装完成"
else
    echo "✓ 插件已存在"
fi

echo -e "${YELLOW}步骤 7/9: 配置 Claude 设置${NC}"
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
    echo "✓ Claude 配置已创建"
else
    echo "✓ Claude 配置已存在"
fi

echo -e "${YELLOW}步骤 8/9: 安装 GSD (Get Shit Done)${NC}"
GSD_INSTALL_DIR="$CONFIG_DIR/get-shit-done"
if [ ! -d "$GSD_INSTALL_DIR" ]; then
    echo "正在克隆 GSD 仓库..."
    if command -v git &> /dev/null; then
        git clone https://github.com/OpenAgentsInc/gsd.git "$GSD_INSTALL_DIR" 2>/dev/null || \
        git clone https://github.com/OpenAgentsInc/get-shit-done.git "$GSD_INSTALL_DIR" 2>/dev/null || \
        echo -e "${YELLOW}警告: 无法克隆 GSD 仓库,跳过${NC}"
    else
        echo -e "${YELLOW}警告: git 未安装,跳过 GSD 安装${NC}"
    fi
    
    if [ -d "$GSD_INSTALL_DIR" ]; then
        echo "✓ GSD 安装完成"
        
        if [ -d "$GSD_INSTALL_DIR/skills" ]; then
            ln -sf "$GSD_INSTALL_DIR/skills" "$CLAUDE_DIR/skills" 2>/dev/null || true
        fi
    fi
else
    echo "✓ GSD 已存在"
fi

echo -e "${YELLOW}步骤 9/9: 创建技能符号链接${NC}"
skill_count=0

for skills_source in "$ORCHESTRA_DIR/skills" "$AGENTS_DIR/skills" "$GSD_INSTALL_DIR/skills"; do
    if [ -d "$skills_source" ]; then
        for skill_dir in "$skills_source"/*; do
            if [ -d "$skill_dir" ]; then
                skill_name=$(basename "$skill_dir")
                target="$CONFIG_DIR/skills/$skill_name"
                if [ ! -e "$target" ]; then
                    ln -s "$skill_dir" "$target" 2>/dev/null && ((skill_count++)) || true
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
echo ""
echo "1. 配置 API 密钥:"
echo "   nano $CONFIG_DIR/opencode.json"
echo "   替换 YOUR_API_KEY_HERE 为你的 Anthropic API 密钥"
echo ""
echo "2. (可选) 调整模型配置:"
echo "   nano $CONFIG_DIR/oh-my-openagent.json"
echo ""
echo "3. 重启 OpenCode 使配置生效"
echo ""
echo -e "${GREEN}配置文件位置:${NC}"
echo "  - OpenCode: $CONFIG_DIR/opencode.json"
echo "  - Agents: $CONFIG_DIR/oh-my-openagent.json"
echo "  - Claude: $CLAUDE_DIR/settings.json"
echo "  - Skills: $CONFIG_DIR/skills/ ($skill_count 个)"
if [ -d "$GSD_INSTALL_DIR" ]; then
    echo "  - GSD: $GSD_INSTALL_DIR"
fi
echo ""
echo -e "${BLUE}提示:${NC}"
echo "  - 如果需要安装更多技能,可以使用 OpenCode 的技能管理功能"
echo "  - GSD 技能提供了强大的项目管理和开发工作流功能"
echo "  - 运行 'opencode skill list' 查看可用技能"
