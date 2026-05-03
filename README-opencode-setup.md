# OpenCode 一键配置工具

这个工具包可以帮助你快速配置 OpenCode 环境,包括 oh-my-openagent 插件和所有相关设置。

## 文件说明

- `setup-opencode.sh` - 主配置脚本,自动设置所有必需的配置文件
- `backup-opencode-config.sh` - 配置备份工具,在修改前备份现有配置

## 使用方法

### 1. 首次配置

```bash
cd /root/vitfly
./setup-opencode.sh
```

脚本会自动:
- 创建必要的配置目录
- 生成 `opencode.json` 配置文件
- 生成 `oh-my-openagent.json` 配置文件
- 安装 oh-my-openagent 插件
- 配置 Claude 设置
- 创建技能符号链接

### 2. 配置 API 密钥

编辑 `~/.config/opencode/opencode.json`:

```bash
nano ~/.config/opencode/opencode.json
```

替换 `YOUR_API_KEY_HERE` 为你的实际 API 密钥。

如果使用代理服务,修改 `baseURL`:
```json
"baseURL": "https://your-proxy-url.com/api/v1"
```

### 3. 备份现有配置(可选)

在运行配置脚本前,可以先备份现有配置:

```bash
./backup-opencode-config.sh
```

备份文件会保存在 `~/opencode-config-backup-YYYYMMDD-HHMMSS/` 目录。

### 4. 重启 OpenCode

配置完成后,重启 OpenCode 使配置生效。

## 配置文件位置

- **OpenCode 主配置**: `~/.config/opencode/opencode.json`
- **Agent 配置**: `~/.config/opencode/oh-my-openagent.json`
- **Claude 配置**: `~/.claude/settings.json`
- **技能目录**: `~/.config/opencode/skills/`

## 自定义配置

### 修改模型配置

编辑 `~/.config/opencode/oh-my-openagent.json` 可以为不同的 agent 和 category 配置不同的模型:

```json
{
  "agents": {
    "oracle": {
      "model": "anthropic/claude-opus-4-6"
    },
    "explore": {
      "model": "anthropic/claude-sonnet-4-6"
    }
  },
  "categories": {
    "ultrabrain": {
      "model": "anthropic/claude-opus-4-6"
    },
    "quick": {
      "model": "anthropic/claude-haiku-4-6"
    }
  }
}
```

### 添加权限

编辑 `~/.config/opencode/opencode.json` 的 `permission` 部分:

```json
"permission": {
  "read": {
    "~/.config/opencode/get-shit-done/*": "allow",
    "~/your-project/*": "allow"
  },
  "external_directory": {
    "~/.config/opencode/get-shit-done/*": "allow"
  }
}
```

## 可用的 Agents

配置文件包含以下专用 agents:

- **hephaestus** - 构建和实现
- **oracle** - 高级推理和架构设计
- **librarian** - 代码库理解和文档检索
- **explore** - 代码库探索和模式发现
- **multimodal-looker** - 多模态内容分析
- **prometheus** - 计划和策略
- **metis** - 计划前咨询
- **momus** - 计划评审
- **atlas** - 知识管理
- **sisyphus-junior** - 专注任务执行

## 可用的 Categories

- **visual-engineering** - 前端、UI/UX、设计
- **ultrabrain** - 高难度逻辑任务
- **deep** - 深度问题解决
- **artistry** - 创造性问题解决
- **quick** - 简单快速任务
- **unspecified-low** - 低工作量任务
- **unspecified-high** - 高工作量任务
- **writing** - 文档和写作

## 故障排除

### 插件安装失败

如果 npm 安装失败,手动安装:

```bash
cd ~/.config/opencode
npm install @opencode-ai/plugin@1.4.6
```

### 技能链接失败

手动创建技能符号链接:

```bash
mkdir -p ~/.config/opencode/skills
ln -s ~/.orchestra/skills/0-autoresearch-skill ~/.config/opencode/skills/autoresearch
```

### 配置不生效

1. 检查 JSON 文件格式是否正确
2. 确保 API 密钥已正确配置
3. 重启 OpenCode
4. 检查日志: `~/.config/opencode/logs/`

## 恢复配置

如果需要恢复之前的配置:

```bash
cp ~/opencode-config-backup-YYYYMMDD-HHMMSS/opencode.json ~/.config/opencode/
cp ~/opencode-config-backup-YYYYMMDD-HHMMSS/oh-my-openagent.json ~/.config/opencode/
cp ~/opencode-config-backup-YYYYMMDD-HHMMSS/settings.json ~/.claude/
```

## 更多信息

- OpenCode 文档: https://opencode.ai/docs
- oh-my-openagent: https://github.com/code-yeongyu/oh-my-openagent
- 问题反馈: 在项目仓库提交 issue
