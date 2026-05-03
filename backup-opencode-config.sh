#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BACKUP_DIR="$HOME/opencode-config-backup-$(date +%Y%m%d-%H%M%S)"

echo -e "${GREEN}=== OpenCode 配置备份工具 ===${NC}"
echo ""

mkdir -p "$BACKUP_DIR"

echo -e "${YELLOW}备份配置文件...${NC}"

if [ -f "$HOME/.config/opencode/opencode.json" ]; then
    cp "$HOME/.config/opencode/opencode.json" "$BACKUP_DIR/"
    echo "✓ opencode.json"
fi

if [ -f "$HOME/.config/opencode/oh-my-openagent.json" ]; then
    cp "$HOME/.config/opencode/oh-my-openagent.json" "$BACKUP_DIR/"
    echo "✓ oh-my-openagent.json"
fi

if [ -f "$HOME/.claude/settings.json" ]; then
    cp "$HOME/.claude/settings.json" "$BACKUP_DIR/"
    echo "✓ Claude settings.json"
fi

if [ -f "$HOME/.config/opencode/package.json" ]; then
    cp "$HOME/.config/opencode/package.json" "$BACKUP_DIR/"
    echo "✓ package.json"
fi

if [ -d "$HOME/.config/opencode/skills" ]; then
    mkdir -p "$BACKUP_DIR/skills"
    ls -la "$HOME/.config/opencode/skills" > "$BACKUP_DIR/skills/links.txt"
    echo "✓ skills 符号链接列表"
fi

echo ""
echo -e "${GREEN}备份完成!${NC}"
echo "备份位置: $BACKUP_DIR"
echo ""
echo "恢复命令:"
echo "  cp $BACKUP_DIR/opencode.json ~/.config/opencode/"
echo "  cp $BACKUP_DIR/oh-my-openagent.json ~/.config/opencode/"
echo "  cp $BACKUP_DIR/settings.json ~/.claude/"
