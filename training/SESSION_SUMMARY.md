# Session Workflow Summary

本文档记录了本项目的开发工作流和关键决策。

## 项目概述

- **项目**: ViT-Fly Mamba Training
- **目标**: 创建最小可运行版本，清理无用文件，整合文档，TDD验证
- **时间**: 2026-04-01

## 完成的工作

### 1. 文件清理

| 类别 | 数量 | 示例 |
|------|------|------|
| Markdown 文件 | 20+ | AGENTS.md, COMPLETE_EXPERIMENT_REPORT.md |
| Python 脚本 | 8+ | check_dataset.py, analyze_dataset.py |
| 缓存文件 | 6+ | __pycache__/* |

### 2. 文档整合

- **根目录**: 单一 `README.md` 包含所有必要信息
- **训练目录**: `training/HANDOFF.md` - 交接文档
- **问题解决**: `training/TROUBLESHOOTING.md` - 故障排除指南

### 3. TDD 验证流程

```
1. Import 测试
   → python -c "import train_mamba_optimized"
   
2. 数据加载测试
   → python -c "from dataloading import dataloader; ..."
   
3. 训练测试（1 epoch）
   → python train_mamba_optimized.py --branches A --epochs 1
   
4. 多分支验证
   → 测试 Branches A, B, C
```

### 4. Git 工作流

```bash
# 1. 检查状态
git status

# 2. 添加更改
git add -A

# 3. 提交
git commit -m "描述"

# 4. 推送（两种方式）

# 方式A: 使用 embeded token
git push https://USER:TOKEN@github.com/REPO.git branch

# 方式B: 使用 gh CLI
echo "TOKEN" | gh auth login --with-token
git push https://github.com/USER/REPO.git branch
```

## 关键决策

### Q: 数据集应该删除吗？
**A**: 不删除。数据集是训练必需的，README 提供了外部下载链接。

### Q: 为什么验证集可能为空？
**A**: 当 trajectory-level split 用于小数据集时。已添加 sample-level split 逻辑作为修复。

### Q: 为什么需要多次验证？
**A**: 第一次验证后发现问题（剩余文件、网络问题），需要迭代修复。Oracle 验证循环确保真正完成。

## 验证结果

| 项目 | 结果 |
|------|------|
| 文件清理 | ✅ 30+ 文件删除 |
| 文档整合 | ✅ 3 个文档（README, HANDOFF, TROUBLESHOOTING） |
| TDD 验证 | ✅ Branches A,B,C 测试通过 |
| GitHub 推送 | ✅ 成功 |
| Val Loss | 3.8230（正常数值，非 inf） |

## 提交历史

```
245d595 Add training handoff documentation
3628113 Final verification
0a45ac9 Verify training works: Val Loss 3.8230
2be1550 Clean all pycache, add TDD verification docs
502557a Verify branches A-C work, clean cache files
acb5536 Final cleanup: remove remaining docs
```

## 使用的命令

```bash
# 快速测试
python train_mamba_optimized.py --branches A --epochs 1 --short 10 --val_split 0.2

# 完整训练
python train_mamba_optimized.py --branches A B C D E --epochs 100 \
  --data_dir /root/vitfly/training/datasets/data_full

# 查看 GPU
nvidia-smi
```

## 经验教训

1. **不要假设** - 必须实际运行训练来验证，不能只看代码
2. **验证循环** - Oracle 多次验证确保真正完成
3. **网络问题** - Git push 可能失败，需要多种方式重试
4. **文档位置** - Handoff 文档在 training/ 而非根目录
5. **处理 inf** - 验证集为空时返回 inf 是预期行为，需要正确处理