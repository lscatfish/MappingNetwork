# Mapping Networks 论文复现 — 设计文档

## 1. 概述

本项目的目标是严格复现论文 *Mapping Networks*（arXiv:2602.19134v1）的核心实验。
通过实现 Mapping Network 架构，在 MNIST 数据集上验证 Single Latent Vector Training (SLVT / Ours\*)
和 Layer-wise Training (LWT / Ours†) 两种训练策略，达到接近或超越基线 CNN1 / CNN2 的分类准确率，
同时实现 50–500× 的参数压缩。

## 2. 项目结构

```
mapping_network/
├── __init__.py
├── target_nets/
│   ├── __init__.py
│   ├── base.py              # TargetNet 抽象基类
│   ├── cnn1.py              # CNN1 (AlexNet 风格)
│   ├── cnn1_3conv.py        # CNN1 三卷积版本
│   └── cnn2.py              # CNN2 (LeNet 风格)
├── mapping/
│   ├── __init__.py
│   └── mapping_net.py       # MappingNetwork(nn.Module)
│   └── loss.py              # MappingLoss
├── trainer/
│   ├── __init__.py
│   ├── slvt.py              # SLVT 训练器
│   └── lwt.py               # LWT 训练器
├── scripts/
│   ├── train.py             # 统一训练入口
│   └── evaluate.py          # 评估入口
└── configs/
    ├── cnn1_slvt.yaml
    ├── cnn1_3conv_slvt.yaml
    ├── cnn1_lwt.yaml
    ├── cnn2_slvt.yaml
    └── cnn2_lwt.yaml
```

## 3. 目标网络架构

### 3.1 CNN2 — LeNet 风格（~108,610 参数）

| Layer | In→Out | Kernel | Output | Params |
|-------|--------|--------|--------|--------|
| Conv1 | 1→20 | 5×5 | 24×24 | 520 |
| AvgPool | 20 | 2×2 | 12×12 | — |
| Conv2 | 20→32 | 5×5 | 8×8 | 16,032 |
| AvgPool | 32 | 2×2 | 4×4 | — |
| Flatten | — | — | 512 | — |
| FC1 | 512→176 | — | — | 90,288 |
| FC2 | 176→10 | — | — | 1,770 |
| **Total** | | | | **108,610** |

激活函数全部用 ReLU，最终输出用 Softmax。

### 3.2 CNN1 — AlexNet 风格（~537,912 参数）

| Layer | In→Out | Kernel | Output | Params |
|-------|--------|--------|--------|--------|
| Conv1 | 1→48 | 5×5 | 24×24 | 1,200 |
| AvgPool | 48 | 2×2 | 12×12 | — |
| Conv2 | 48→128 | 5×5 | 8×8 | 153,728 |
| AvgPool | 128 | 2×2 | 4×4 | — |
| Flatten | — | — | 2,048 | — |
| FC1 | 2,048→186 | — | — | 381,114 |
| FC2 | 186→10 | — | — | 1,870 |
| **Total** | | | | **537,912** |

### 3.3 CNN1-3Conv — AlexNet 风格三卷积版（实验性）

作为额外实验保留，架构待实现时根据参数量匹配调整。
主实验以 3.2 节两层卷积版本为准。

## 4. Mapping Network 核心

### 4.1 架构

```
Trainable Latent Vector z ∈ R^d
        │
        ▼
MappingNetwork (fixed weights W ∈ R^{P×d}, bias b ∈ R^P)
  W 经调制: w_ij ← w_ij + α·z_i   (方程 20)
        │
        ▼
θ̂ = σ(W_modulated · z + b) ∈ R^P  (方程 21)
        │
        ▼  reshape + 切分
Target Network 各层参数 {W^(l), b^(l)}
        │
        ▼
Target Network forward (不训练, 只前向)
```

### 4.2 函数式前向（替代 .data.copy_()）

**关键设计决策**：不使用 `.data.copy_()` 注入参数（会切断 autograd 梯度链），
而是让 TargetNet 实现**函数式前向** `functional_forward(x, theta_hat, slices)`：

```
θ̂ 按累积偏移量切分并 reshape 为各层 weight/bias tensor
        │
        ▼
F.conv2d(x, weight, bias)   ←  直接使用 θ̂ 切片作为权重参数
F.linear(x, weight, bias)   ←  autograd 自然追踪到 θ̂ → W_mod → z
        │
        ▼
梯度路径: L_map → functional_forward → θ̂ 切片 → W_modulated → z ✓
（MappingNet 的 W, b 固定，不参与梯度）
```

每个 TargetNet 子类实现 `_functional_forward(x, params_dict)`，
在其中用 `F.conv2d` / `F.linear` 等函数式 API 替代 `self.conv1(x)` 等模块调用。
`params_dict` 的键为 `'conv1.weight'`、`'fc1.bias'` 等，值从 θ̂ 切片 reshape 得到。

这一方案同时解决了 L_stab 噪声计算中 save/restore 参数的问题——
无需修改目标网络参数，直接传入 `theta_hat` 和 `theta_noisy` 即可分别前向。

## 5. Mapping Loss

L_map = L_task + λ_st·L_stab + λ_sm·L_smooth + λ_al·L_align  (方程 26)

| 项 | 公式 | 作用 |
|----|------|------|
| L_task | CrossEntropy(ŷ, y) | 分类任务损失 |
| L_stab | E[∥f(z+ε)−f(z)∥²], ε~N(0,σ²I) | Lipschitz 连续性 |
| L_smooth | ∥∇_z M_φ(z)∥²_F / P | C² 光滑性（按 P 平均，跨实验可迁移） |
| L_align | 1−cos(z, W_m) | 对齐 latent 与权重方向 |

λ_st, λ_sm, λ_al 为可训练的 nn.Parameter，初始值 0.1。
各损失项统一使用 .backward() 回传梯度至 z，MappingNet 权重固定不更新。

## 6. 训练策略

### 6.1 SLVT — Single Latent Vector Training (Ours\*)

- 一个 `z` 生成全部目标网络参数
- latent 维度 d = 1024, 2048, 2072, 4078 (按实验配置)
- 映射矩阵形状：P × d

### 6.2 LWT — Layer-wise Training (Ours†)

- 每层/每组有独立 `z^(l)` 和对应的映射网络
- 每个 layer 的 latent 维度可独立配置
- 各层调制率 α^(l) 独立

### 6.3 训练循环

SLVT 训练循环：

```
for epoch in epochs:
  for x, y in dataloader:
    θ̂ = mapping_net(z)                        # 生成参数 [P]
    ŷ = target_net.functional_forward(x, θ̂, slices)  # 函数式前向，梯度完整
    loss, losses = mapping_loss(z, θ̂, mapping_net, target_net, x, y, slices)
    loss.backward()                           # 梯度回传至 z (和 λ)
    optimizer.step()                          # 只更新 z 和 λ
    optimizer.zero_grad()
```

LWT 训练循环（每层独立 z^(l) + MappingNet^(l)）：

```
θ̂ = 拼接各 layer_mapping 的 θ̂^(l)
ŷ = target_net.functional_forward(x, θ̂, slices)
loss = L_task(ŷ, y) + Σ_l λ·(L_stab(z^(l)) + L_smooth(z^(l)) + L_align(z^(l)))
loss.backward()
optimizer.step()    # 更新所有 z^(l) 和 λ
```

### 6.4 优化器与调度器

- 优化器：**AdamW**（权重衰减 1e-4）
- 学习率调度器：**余弦退火**（Cosine Annealing, T_max = epochs）
  - 初始学习率：1e-3
  - 最小学习率：1e-5
- Batch size：64 或 128
- Epochs：按需（论文 20–50 epoch 量级）
- 可选项：论文未明确指定优化器，亦可尝试论文常见配置 Adam
  无调度器作为对比基线

### 6.5 Checkpoint 保存

- SLVT: 训练结束时保存 `mapping_net.state_dict()` 到 `checkpoints/slvt_{target_net}_{latent_dim}.pth`
- LWT: 保存所有 `layer_mappings` 的 state_dict，结构为 `{layer_name: state_dict}`，到 `checkpoints/lwt_{target_net}.pth`
- 同时保存 `results`（含每个 epoch 的 loss/acc）到同目录 `.json` 文件

### 6.6 随机种子

- 在训练入口设置 `torch.manual_seed`、`torch.cuda.manual_seed_all`
- 配置文件中可指定 `seed: 42`

## 7. 实验结果预期复现表

### CNN2 实验复现目标

| Method | \#Params | MNIST | FMNIST |
|--------|----------|-------|--------|
| CNN2 (baseline) | 108,618 | 98.69% | 90.40% |
| Ours\* | 1,024 | 97.88% | 89.49% |
| Ours\* | 2,048 | 98.66% | 91.88% |
| Ours† | 1,872 | 98.98% | 92.84% |
| Ours† | 2,688 | 99.18% | 93.35% |

### CNN1 实验复现目标

| Method | \#Params | MNIST | FMNIST |
|--------|----------|-------|--------|
| CNN1 (baseline) | 537,994 | 99.32% | 92.89% |
| Ours\* | 1,024 | 98.78% | 93.02% |
| Ours\* | 2,072 | 99.56% | 93.91% |
| Ours† | 4,078 | 99.67% | 94.83% |

## 8. 配置文件样例

```yaml
# configs/cnn2_slvt.yaml
target_net: cnn2
training_strategy: slvt
latent_dim: 2048
batch_size: 64
epochs: 30
seed: 42

# Optimizer & Scheduler
optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine_annealing
min_lr: 0.00001

# Mapping Network
alpha: 0.01

# Mapping Loss
lambda_st_init: 0.1
lambda_sm_init: 0.1
lambda_al_init: 0.1
sigma_noise: 0.01

device: cuda
log_interval: 100
checkpoint_dir: checkpoints
```

## 9. 分支策略

- 从 `main` 创建新分支 `feat/mapping-network-reproduction`
- 迭代开发：先实现 TargetNets → MappingNetwork → MappingLoss → 训练器 → 实验脚本
