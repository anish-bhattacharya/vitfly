#!/bin/bash
# OpenCode 一键配置脚本
# 自动配置 OpenCode 环境，包括 oh-my-openagent 插件和所有相关设置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OpenCode 一键配置脚本 ===${NC}"
echo ""

# 检查是否在 OpenCode 环境中
if [ -z "$OPENCODE" ]; then
    echo -e "${RED}错误: 未检测到 OpenCode 环境${NC}"
    echo "请在 OpenCode 终端中运行此脚本"
    exit 1
fi

# 配置目录
CONFIG_DIR="$HOME/.config/opencode"
CLAUDE_DIR="$HOME/.claude"
ORCHESTRA_DIR="$HOME/.orchestra"
AGENTS_DIR="$HOME/.agents"

echo -e "${YELLOW}步骤 1/6: 创建配置目录${NC}"
mkdir -p "$CONFIG_DIR"
mkdir -p "$CLAUDE_DIR"
mkdir -p "$ORCHESTRA_DIR/skills"
mkdir -p "$AGENTS_DIR/skills"

echo -e "${YELLOW}步骤 2/6: 配置 opencode.json${NC}"
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

echo -e "${YELLOW}步骤 3/6: 配置 oh-my-openagent.json${NC}"
cat > "$CONFIG_DIR/oh-my-openagent.json" << 'EOF'
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-openagent/dev/assets/oh-my-opencode.schema.json",
  "agents": {
    "hephaestus": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "oracle": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "librarian": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "explore": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "multimodal-looker": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "prometheus": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "metis": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "momus": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "atlas": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "sisyphus-junior": {
      "model": "anthropic/claude-sonnet-4-6"
    }
  },
  "categories": {
    "visual-engineering": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "ultrabrain": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "deep": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "artistry": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "quick": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "unspecified-low": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "unspecified-high": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "writing": {
      "model": "anthropic/claude-sonnet-4-6"
    }
  }
}
EOF

echo -e "${YELLOW}步骤 4/6: 安装 oh-my-openagent 插件${NC}"
cd "$CONFIG_DIR"
if [ ! -d "node_modules" ]; then
    npm install @opencode-ai/plugin@1.4.6
fi

echo -e "${YELLOW}步骤 5/6: 配置 Claude 设置${NC}"
cat > "$CLAUDE_DIR/settings.json" << 'EOF'
{
  "hooks": {
    "SessionStart": [],
    "PostToolUse": [],
    "PreToolUse": []
  }
}
EOF

echo -e "${YELLOW}步骤 6/6: 创建技能符号链接${NC}"
mkdir -p "$CONFIG_DIR/skills"
if [ -d "$ORCHESTRA_DIR/skills" ]; then
    for skill_dir in "$ORCHESTRA_DIR/skills"/*; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            if [ ! -e "$CONFIG_DIR/skills/$skill_name" ]; then
                ln -s "$skill_dir" "$CONFIG_DIR/skills/$skill_name"
            fi
        fi
    done
fi

echo ""
echo -e "${GREEN}=== 配置完成 ===${NC}"
echo ""
echo -e "${YELLOW}下一步操作:${NC}"
echo "1. 编辑 $CONFIG_DIR/opencode.json"
echo "   - 替换 YOUR_API_KEY_HERE 为你的 Anthropic API 密钥"
echo "   - 如果使用代理,修改 baseURL"
echo ""
echo "2. 根据需要调整模型配置:"
echo "   - 编辑 $CONFIG_DIR/oh-my-openagent.json"
echo "   - 可以为不同的 agent 和 category 配置不同的模型"
echo ""
echo "3. 重启 OpenCode 使配置生效"
echo ""
echo -e "${GREEN}配置文件位置:${NC}"
echo "  - OpenCode: $CONFIG_DIR/opencode.json"
echo "  - Agents: $CONFIG_DIR/oh-my-openagent.json"
echo "  - Claude: $CLAUDE_DIR/settings.json"
echo "  - Skills: $CONFIG_DIR/skills/"
