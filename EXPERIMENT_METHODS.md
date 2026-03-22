# DroneMamba 实验方法详细说明

## 📚 学习方法类型

### ✅ 模仿学习（Imitation Learning）- 行为克隆

DroneMamba **完全遵循 VitFly 的学习方法**，使用**模仿学习**而非强化学习。

#### 具体方法：行为克隆（Behavioral Cloning）

```python
# 训练过程（来自 training/train.py）
def training_loop(self):
    for epoch in range(N_eps):
        for batch in train_loader:
            # 1. 从专家演示中获取数据
            depth_images = batch.images      # 深度图像
            expert_commands = batch.velcmd   # 专家速度命令（标签）
            
            # 2. 前向传播预测动作
            predicted_commands = model(depth_images)
            
            # 3. 最小化预测与专家动作的差异
            loss = MSE_loss(predicted_commands, expert_commands)
            
            # 4. 反向传播优化参数
            loss.backward()
            optimizer.step()
```

### 与强化学习的区别

| 特性 | **模仿学习 (DroneMamba)** | 强化学习 (RL) |
|-----|------------------------|-------------|
| **数据来源** | 专家演示数据集 | 环境交互试错 |
| **监督信号** | 专家动作（有监督） | 奖励函数（无监督） |
| **训练方式** | 行为克隆（MSE 损失） | 策略梯度/Q-learning |
| **收敛速度** | 快（直接学习映射） | 慢（需要大量探索） |
| **稳定性** | 高（ supervised learning） | 低（方差大） |
| **计算需求** | 低（单次前向传播） | 高（需要 rollout） |
| **典型算法** | Behavior Cloning | PPO, DQN, SAC |

### 专家策略来源

VitFly/DroneMamba 使用的专家数据来自：

1. **基于规则的专家策略**（仿真环境）
   ```python
   # envtest/ros/user_code.py
   def compute_command_expert(state, obstacles, desiredVel):
       # 简化的特权专家策略
       # 1. 检测障碍物位置
       # 2. 计算避障方向
       # 3. 输出速度命令
       return velocity_command
   ```

2. **数据采集流程**
   ```bash
   # 在仿真中运行专家策略收集数据
   roslaunch envsim data_collection.launch
   
   # 生成数据集
   training/datasets/data/
   ├── depth_images/  # 专家看到的深度图
   └── data.csv       # 专家执行的动作
   ```

### 为什么选择模仿学习？

✅ **优势**:
- **样本效率高**: 直接从专家演示学习，不需要大量试错
- **训练稳定**: 标准的监督学习，收敛性好
- **计算成本低**: 单次前向传播即可更新
- **适合避障任务**: 专家策略容易设计（几何避障）

⚠️ **局限性**:
- **依赖专家质量**: 专家策略的性能上限
- **分布外泛化**: 遇到训练未见场景可能失效
- **无法超越专家**: 只能模仿，不能改进

---

## 🔬 消融实验设计

### 什么是消融实验？

消融实验（Ablation Study）通过**系统性地移除或替换模型的某些组件**，来验证每个组件的贡献和必要性。

### DroneMamba 消融实验方案

#### 实验 1: SSM vs Transformer vs CNN

**目的**: 验证 Mamba 架构的有效性

| 变体 | 架构配置 | 预期结果 |
|-----|---------|---------|
| **完整模型** | CNN + SSM (2 层) | 基线（最佳） |
| Variant A | CNN + Transformer (2 层) | 参数量↑，速度↓ |
| Variant B | CNN only (无 SSM/Transformer) | 性能↓，感受野受限 |
| Variant C | SSM only (无 CNN) | 性能↓，缺少局部特征 |

**实现方法**:
```python
# models/model.py 中修改
class DroneMamba_Ablation(nn.Module):
    def __init__(self, ablation_type='full'):
        if ablation_type == 'transformer':
            # 使用 ViT 的 MixTransformerEncoderLayer
            self.encoder_blocks = nn.ModuleList([
                MixTransformerEncoderLayer(...),  # 替换 SSM
                MixTransformerEncoderLayer(...)
            ])
        elif ablation_type == 'cnn_only':
            # 移除 SSM，只用 CNN
            self.encoder_blocks = nn.Identity()
        elif ablation_type == 'ssm_only':
            # 移除初始 CNN
            self.conv1 = nn.Identity()
            self.conv2 = nn.Identity()
```

#### 实验 2: SSM 层数影响

**目的**: 确定最优网络深度

| 变体 | SSM 层数 | 参数量 | 预期性能 |
|-----|---------|--------|---------|
| 1-layer | 1 | 0.25M | 性能↓（欠拟合） |
| **2-layer** | **2** | **0.45M** | **最佳** |
| 3-layer | 3 | 0.65M | 性能→（边际效益） |
| 4-layer | 4 | 0.85M | 性能↓（过拟合） |

#### 实验 3: 状态维度 d_state

**目的**: 分析 SSM 表达能力

| d_state | 参数量 | FLOPs | 预期结果 |
|---------|--------|-------|---------|
| 4 | 0.30M | 低 | 性能↓（容量不足） |
| **8** | **0.45M** | **中** | **最佳** |
| 16 | 0.60M | 高 | 性能→（收益递减） |
| 32 | 0.90M | 很高 | 性能↓（过拟合） |

#### 实验 4: 双向扫描 vs 单向扫描

**目的**: 验证双向扫描的必要性

```python
# mamba_submodules.py
class SimplifiedSSM(nn.Module):
    def __init__(self, bidirectional=True):
        self.bidirectional = bidirectional  # 消融时设为 False
    
    def forward(self, x, H, W):
        if self.bidirectional:
            y_fwd = self.ssm_scan_forward(...)
            y_bwd = self.ssm_scan_backward(...)
            y = torch.cat([y_fwd, y_bwd], dim=-1)
        else:
            y = self.ssm_scan_forward(...)  # 只用前向
```

| 变体 | 扫描方式 | 信息流 | 预期性能 |
|-----|---------|--------|---------|
| **Bidirectional** | 前向 + 后向 | 全局 | **最佳** |
| Unidirectional | 仅前向 | 因果 | 性能↓ |

#### 实验 5: 时序建模模块对比

**目的**: 比较 LSTM vs SSM 用于时序建模

| 变体 | 时序模块 | 参数量 | FPS | 预期成功率 |
|-----|---------|--------|-----|-----------|
| **LSTM** | 2 层 LSTM (hidden=128) | 0.64M | 371 | **90%** |
| **SSM** | Temporal SSM (d_state=4) | 0.45M | 452 | **88%** |
| No Temporal | 无时序模块 | 0.40M | 480 | 80%↓ |

#### 实验 6: 输入模态消融

**目的**: 分析各输入传感器的贡献

| 变体 | 深度图 | 期望速度 | 四元数 | 预期性能 |
|-----|-------|---------|-------|---------|
| **Full** | ✅ | ✅ | ✅ | **最佳** |
| w/o vel | ✅ | ❌ | ✅ | 性能↓↓ |
| w/o quat | ✅ | ✅ | ❌ | 性能↓ |
| vision only | ✅ | ❌ | ❌ | 性能↓↓↓ |

### 消融实验执行流程

#### 步骤 1: 创建消融模型

在 `models/model.py` 中添加消融变体：

```python
class DroneMambaAblation(DroneMamba):
    """DroneMamba 消融实验版本"""
    
    def __init__(self, 
                 ablation_ssm=False,      # 移除 SSM
                 ablation_cnn=False,      # 移除 CNN
                 ablation_temporal=False, # 移除时序模块
                 ssm_layers=2,            # SSM 层数
                 d_state=8,               # 状态维度
                 bidirectional=True):     # 双向扫描
        super().__init__()
        
        # 根据消融配置修改网络结构
        if ablation_ssm:
            self.mamba_block1 = nn.Identity()
            self.mamba_block2 = nn.Identity()
        
        if ablation_cnn:
            self.conv1 = nn.Identity()
            self.conv2 = nn.Identity()
        
        # ... 其他消融配置
```

#### 步骤 2: 训练所有变体

创建训练脚本 `train_ablations.sh`:

```bash
#!/bin/bash

# 完整模型
python3 train.py --config config/train_mamba.txt \
  --model_name DroneMamba_Full

# 消融 1: 移除 SSM
python3 train.py --config config/train_mamba.txt \
  --model_name DroneMamba_NoSSM \
  --ablation_ssm

# 消融 2: 移除 CNN
python3 train.py --config config/train_mamba.txt \
  --model_name DroneMamba_NoCNN \
  --ablation_cnn

# 消融 3: 1 层 SSM
python3 train.py --config config/train_mamba.txt \
  --model_name DroneMamba_1Layer \
  --ssm_layers 1

# ... 其他消融实验
```

#### 步骤 3: 评估并对比

创建评估脚本 `evaluate_ablations.py`:

```python
import pandas as pd

results = []

# 评估所有消融变体
for model_name in ['Full', 'NoSSM', 'NoCNN', '1Layer', '2Layer', '3Layer']:
    metrics = evaluate_model(f'DroneMamba_{model_name}')
    results.append({
        'Model': model_name,
        'Params': metrics['params'],
        'FPS': metrics['fps'],
        'Success Rate': metrics['success_rate'],
        'Loss': metrics['loss']
    })

# 生成对比表格
df = pd.DataFrame(results)
print(df.to_string(index=False))

# 保存结果
df.to_csv('ablation_results.csv', index=False)
```

#### 步骤 4: 可视化结果

创建绘图脚本 `plot_ablations.py`:

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取结果
df = pd.read_csv('ablation_results.csv')

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 参数量对比
axes[0, 0].bar(df['Model'], df['Params'])
axes[0, 0].set_ylabel('Parameters (M)')
axes[0, 0].tick_params(rotation=45)

# 成功率对比
axes[0, 1].bar(df['Model'], df['Success Rate'])
axes[0, 1].set_ylabel('Success Rate (%)')
axes[0, 1].tick_params(rotation=45)

# FPS 对比
axes[1, 0].bar(df['Model'], df['FPS'])
axes[1, 0].set_ylabel('FPS')
axes[1, 0].tick_params(rotation=45)

# 损失对比
axes[1, 1].bar(df['Model'], df['Loss'])
axes[1, 1].set_ylabel('Loss')
axes[1, 1].tick_params(rotation=45)

plt.tight_layout()
plt.savefig('ablation_comparison.png', dpi=300)
plt.show()
```

### 预期消融结果示例

```
=================================================================
Ablation Study Results
=================================================================
Model              Params(M)   FPS   Success(%)   Loss
-----------------------------------------------------------------
DroneMamba_Full       0.45    452      88.5      0.0015  ← 最佳
DroneMamba_NoSSM      0.15    520      75.2      0.0035  ← SSM 重要
DroneMamba_NoCNN      0.35    480      80.1      0.0025  ← CNN 重要
DroneMamba_1Layer     0.30    460      82.3      0.0022  ← 深度不足
DroneMamba_2Layer     0.45    452      88.5      0.0015  ← 最佳深度
DroneMamba_3Layer     0.65    440      88.8      0.0014  ← 收益递减
DroneMamba_UniDir     0.45    455      85.2      0.0018  ← 双向更好
=================================================================

结论:
✓ SSM 模块贡献：+13.3% 成功率提升
✓ CNN 模块贡献：+8.4% 成功率提升  
✓ 2 层 SSM 最优：性能和效率平衡
✓ 双向扫描：+3.3% 成功率提升
```

---

## 📊 完整实验计划

### Phase 1: 主实验（1-2 周）

1. **训练 DroneMamba（完整模型）**
   ```bash
   ./run_full_experiment.sh --epochs 60
   ```

2. **与基线模型对比**
   - ViT
   - ViT+LSTM
   - LSTMNet
   - ConvNet

3. **评估指标**
   - 成功率
   - 平均飞行距离
   - 碰撞率
   - 推理速度

### Phase 2: 消融实验（2-3 周）

1. **架构消融**（实验 1-3）
2. **时序建模消融**（实验 4-5）
3. **输入模态消融**（实验 6）

### Phase 3: 分析与论文撰写（1-2 周）

1. **结果分析**
2. **图表生成**
3. **论文撰写**

---

## 💡 关键要点总结

### 学习方法
✅ **模仿学习（行为克隆）** - 非强化学习
- 从专家演示学习
- MSE 损失最小化
- 监督学习范式

### 消融实验设计原则
1. **控制变量**: 每次只改变一个组件
2. **公平对比**: 保持训练配置一致
3. **多次运行**: 减少随机性影响
4. **统计显著**: 报告均值±标准差

### 预期贡献
1. **方法创新**: 简化 SSM 设计
2. **效率提升**: 参数量减少 85%+
3. **速度优势**: 推理速度快 2.6x
4. **实证分析**: 系统性消融研究

---

*最后更新：2026-03-16*
