# Mapping 框架扩展与 mapping_network 重构 - 设计文档

> 日期: 2026-07-19
> 状态: 设计阶段（已通过用户评审）
> 前置文档: `2026-07-19-mapping-framework-design.md`（三层架构基础）

## 1. 概述

### 1.1 目标

在已实现的 mapping 三层框架（generator 子块 → Generator → MappingLayer/Sequential）之上：

1. **改进 generator 子块的可组合性**：新增 `Block` 基类，用户像继承 `nn.Module` 一样继承它即可自由组合（残差块等），参数冻结与初始化自动完成
2. **扩展更丰富的功能**：预置积木块、更多主干层类型、`generator_instance` 传入
3. **用新框架重构 `mapping_network/` 旧功能**：MappingLoss、Trainer、CNN 目标网络、训练/评估脚本（Python 配置，弃用 yaml）
4. **旧包分阶段废弃**：先标记 deprecated，新功能跑通后删除

### 1.2 设计原则

沿用前序设计原则，新增：

6. **继承即所得**：组合积木只需继承 `Block`，init 与 forward 写法与 torch 完全一致，不引入 `build()` 等非 torch 概念
7. **损失/训练通用化**：`MappingLoss` 与 `Trainer` 只依赖 `generator.z` 与 `generator.forward()`，不要求用户 generator 实现额外方法
8. **配置即 Python**：训练脚本用 argparse + Python 配置，不再使用 yaml

---

## 2. generator 积木基类（`mapping.generator.Block`）

### 2.1 机制

新增 `Block` 基类，通过元类在用户 `__init__` 结束后自动执行两步：

1. 调用 `self.init_weights()`（用户可重载；默认对叶子块做论文初始化，组合块默认不重初始化子块）
2. 递归冻结全部参数：`param.requires_grad_(False)`

用户写法与 torch 完全一致：

```python
class ResBlock(mapping.generator.Block):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = mapping.generator.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = mapping.generator.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))
```

不需要手动 freeze、不需要手动调 init——继承 `Block` 即得。组合块嵌套时，外层 `Block` 的冻结覆盖所有后代参数（幂等，叶子块已冻结再冻一次无副作用）。

### 2.2 现有子块重构

`generator.Linear / Conv1d / Conv2d` 重构为 `Block` 子类，对外签名与行为不变（论文初始化 + `requires_grad=False` + 可重载 `init_weights`）。

`LRDLayer` 保持原样（无参辅助模块，不需要 Block 机制）。

### 2.3 预置积木块（`mapping.generator`）

全部基于 `Block` 实现，与用户自定义写法同一套机制：

- `generator.MLP(sizes, act=nn.ReLU)`：多层 Linear + 激活
- `generator.ResBlock`：残差块（linear 版与 conv 版，跳连相加）
- `generator.TransformerBlock(dim, num_heads, ...)`：标准 pre-norm Transformer 块（内部注意力/FFN 参数固定）

### 2.4 `generator_instance` 扩展

`MappingLayer` 新增 `generator_instance=` 参数：

- 接受已实例化的 `Generator`，构造时校验其 `param_spec` 与层自动推导的一致，不一致抛 `ValueError`
- 与 `generator_cls` 互斥：两者同传抛 `ValueError`
- 语义 = **权重捆绑**：挂同一实例的多个层获得完全相同的 `(weight, bias)`
- 注意与 Sequential（SLVT，切片共享）的区分：部分层各拿不同切片用嵌套 `Sequential` 表达；完全相同权重才用 `generator_instance`

---

## 3. 主干层扩展（`mapping.layers`）

与现有 `Conv2d`/`Linear` 同模式（param_spec 自动推导 + 函数式前向 + `generator_cls`/`generator_instance`/`**generator_kwargs`）：

- `mapping.Conv1d`：`F.conv1d`，param_spec `weight: (C_out, C_in, k)`
- `mapping.ConvTranspose2d`：`F.conv_transpose2d`，param_spec `weight: (C_in, C_out, kh, kw)`（注意 torch 转置卷积权重布局）
- `mapping.BatchNorm1d / BatchNorm2d`：`weight`/`bias` 由 generator 生成；`running_mean`/`running_var`/`num_batches_tracked` 作为 buffer 保留在层内，前向用 `F.batch_norm`
- `mapping.ResBlock`（主干级）：两个 `mapping.Conv2d` + 跳连的 `MappingLayer` 容器
  - LWT 模式：内部各层各自带 generator
  - SLVT 模式：作为纯形状层放入 `Sequential`。param_spec 为**聚合 flat**：`weight: (total_w,)`（内部所有参数层权重按声明顺序拼接的 1D 大小）、`bias: (total_b,)` 同理；`forward_with_params` 收到整段切片后内部按边界二次切片，分发给各内部层——对外仍满足现有「每层一对 (weight, bias)」的供参模型，Sequential 无需改动
  - 通道数或空间尺寸变化时自动启用 1×1 shortcut 卷积（其权重同样计入聚合 flat）

---

## 4. `mapping.loss`（MappingLoss 迁移）

从 `mapping_network/mapping/loss.py` 迁移并通用化。只依赖 `generator.z`（`nn.Parameter`）与 `generator.forward() -> (weight, bias)`，不要求用户 generator 实现额外方法：

- `L_task`：交叉熵（接收主干网络 logits 与 target）
- `L_stab`：采样 `n_stab_samples` 次 `z + σ·ε` 重新调用 `generator.forward()`，与无噪输出求 MSE
- `L_smooth`：`‖∇_z M(z)‖²_F / (P·d)`，用 `torch.autograd.functional.jacobian` 对 `z` 求导
- `L_align`：`1 − cos(z, mean(生成的 weight))`
- 总损失：`L_task + sigmoid(λ_st)·L_stab + sigmoid(λ_sm)·L_smooth + sigmoid(λ_al)·L_align`，λ 为可学习参数（沿用旧设计）
- 双模式：
  - LWT：对每层 generator 独立计算 `L_smooth`/`L_align`/`L_stab` 并聚合（对齐旧 `LWTTrainer` 的逐层正则 + `theta.detach()` 防跨层泄漏语义）
  - SLVT：对共享 generator 整体计算

---

## 5. `mapping.trainer`（训练循环迁移）

- `BaseTrainer`：epoch 训练循环、评估循环、checkpoint 保存/加载、日志
- `LWTTrainer`：遍历主干网络中各 `MappingLayer` 的 `generator`，构建参数组（所有 z + 所有 λ），逐层正则损失
- `SLVTTrainer`：以 `net.generator` 为参数来源，整体正则损失
- `optim_utils`：optimizer / scheduler 工厂，改由 Python 配置驱动（dict 或 dataclass）

---

## 6. `examples/`（实验代码）

新建顶层 `examples/` 目录，定位为示例代码：

```
examples/
    models/           # CNN1 / CNN2 / CNN1_3Conv 用新框架重写
        cnn1.py       # LWT 版 + SLVT 版 + baseline（纯 torch）
        cnn2.py
        cnn1_3conv.py
    train.py          # mapping 训练入口（argparse + Python 配置）
    train_baseline.py # baseline 训练入口
    evaluate.py       # 评估入口
    data.py           # MNIST 加载（自 mapping_network/data.py 迁移，沿用 data/ 目录）
```

- 三个目标网络结构对齐旧实现（卷积/池化/全连接配置一致），保证实验可比
- 不再使用 yaml；训练超参（lr、epochs、λ 初值、n_stab_samples、z_dim 等）用 Python 配置对象表达

---

## 7. 旧包分阶段废弃

- **阶段 7a（随首个迁移阶段开始）**：`mapping_network/` 顶层 `__init__.py` 加 `DeprecationWarning`，README 标注指向新框架；不改动其内部逻辑
- **阶段 7b（examples 全部跑通、实验结果可比后）**：删除整个 `mapping_network/` 包

---

## 8. 阶段拆分（对应 GitHub issue）

| # | 阶段 | 内容 | 依赖 |
|---|------|------|------|
| 1 | generator 积木基类 | `Block` 元类基类 + 现有子块重构 + `generator_instance` 扩展 | — |
| 2 | 预置 generator 块 | `generator.MLP` / `generator.ResBlock` / `generator.TransformerBlock` | 1 |
| 3 | 主干层扩展 | `Conv1d` / `ConvTranspose2d` / `BatchNorm1d/2d` / `mapping.ResBlock` | — |
| 4 | mapping.loss | MappingLoss 通用化迁移（LWT/SLVT 双模式） | — |
| 5 | mapping.trainer | BaseTrainer / LWTTrainer / SLVTTrainer / optim_utils | 4 |
| 6 | examples | CNN1/CNN2/CNN1_3Conv 重写 + train/train_baseline/evaluate 脚本 | 3, 5 |
| 7 | 旧包废弃 | 7a deprecation 标记（可随阶段 1 并行）；7b 删除 mapping_network | 6 |

每阶段沿用 subagent-driven-development 流程与测试规范（GPU fixture、复用业务逻辑、`.venv/bin/python -m pytest`、禁改 `mapping_network/` 内部逻辑）。

---

## 9. 不在范围内

- 多 GPU / 分布式训练
- 新实验设计（只迁移既有实验，保证结果可比）
- baseline 与 mapping 的精度调优
