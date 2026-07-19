# Mapping 推理框架 - 设计文档

> 日期: 2026-07-19
> 状态: 设计阶段

## 1. 概述

### 1.1 定位

Mapping 是一个**纯推理框架**，用于将参数生成网络（Mapping Network）与主干网络组合为可训练的模型。框架只提供前向推理所需的核心原语，所有损失函数、训练循环、z 采样等由用户自行实现。

### 1.2 范围

**框架负责：**
- `mapping.Generator` 基类：可组合的参数生成网络，强制声明 `z_dim`
- `mapping.generator.*` 子块：固定随机参数的积木模块（对齐 torch 的 Linear/Conv1d/Conv2d 签名）
- `mapping.MappingLayer` 基类及具体层（`Conv2d`, `Linear`）：参数规格自动推导，支持 LWT/SLVT 双模式
- `mapping.Sequential`：SLVT 模式的共享 generator 容器
- `mapping.generator.LRDLayer`：低秩分解辅助模块，供用户在 generator 内部使用

**框架不负责（用户自行实现）：**
- 所有损失函数（L_stab, L_smooth, L_align, task loss 等）
- z 的管理与高斯采样
- 训练循环、优化器配置、Trainer 类
- 与现有旧代码的兼容

### 1.3 设计原则

1. **像 torch 一样简洁**：API 风格对齐 PyTorch，init 签名与 `nn.Conv2d`/`nn.Linear` 保持一致
2. **weight/bias 永远分离**：不合并成单一 flat 张量，用户可分别处理
3. **默认 shaped**：generator 输出张量形状严格匹配前向算子（`F.conv2d`/`F.linear`）的输入形状
4. **逐层独立，原生支持 LWT**：每层自带 generator 即为 LWT 模式
5. **Sequential 即 SLVT**：一个共享 generator 管理容器内所有参数层的参数

---

## 2. 三层架构

框架采用严格的三层分离：

```
┌──────────────────────────────────────────────────────────┐
│ Layer 3: MappingLayer (mapping.Conv2d / mapping.Linear)  │
│  - 声明层形状，自动推导 param_spec                          │
│  - forward() 调用自己的 generator → F.conv2d / F.linear   │
│  - forward_with_params() 接收外部参数（SLVT 用）            │
├──────────────────────────────────────────────────────────┤
│ Layer 2: Generator (mapping.Generator)                   │
│  - 强制声明 z_dim，自动创建 self.z (nn.Parameter)          │
│  - 基类自动派生 w_shape/b_shape/w_size/b_size                │
│  - forward() → tuple (weight, bias)                        │
├──────────────────────────────────────────────────────────┤
│ Layer 1: Generator Sub-blocks (mapping.generator.*)      │
│  - 固定随机参数，requires_grad=False                       │
│  - 无 z 强制要求，可复用积木                                │
│  - 默认论文初始化，用户可重载 init_weights                   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: Generator 子块 (`mapping.generator.*`)

### 3.1 定位

可复用的固定随机参数模块，对齐 torch 的 init 签名。内部参数在构造时随机初始化并设为 `requires_grad=False`，**无 z 强制要求**。用户可像搭积木一样自由组合。

### 3.2 提供的子块

```python
mapping.generator.Linear(in_features, out_features, bias=True)
mapping.generator.Conv1d(in_channels, out_channels, kernel_size, stride=1,
                          padding=0, dilation=1, groups=1, bias=True)
mapping.generator.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                          padding=0, dilation=1, groups=1, bias=True)
```

### 3.3 行为

- 构造时内部参数用**论文方法**随机初始化，`requires_grad=False`
- 用户可重载 `init_weights()` 方法自定义初始化
- `forward(x)` 行为与对应 torch 算子一致（`F.linear` / `F.conv1d` / `F.conv2d`）
- 可放入 `nn.Sequential`，可与 `nn.ReLU` 等混合使用

### 3.4 示例

```python
# 作为子块组合
block = nn.Sequential(
    mapping.generator.Linear(64, 128),
    nn.ReLU(),
    mapping.generator.Linear(128, 256),
    nn.ReLU(),
)

# 用户自定义初始化
class MyLinear(mapping.generator.Linear):
    def init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
```

---

## 4. Layer 2: Generator (`mapping.Generator`)

### 4.1 定位

**输入到 MappingLayer 的生成网络**。基类强制 `z_dim`，自动创建 `self.z`。接收 `param_spec`（由 MappingLayer 自动推导并传入），内部组合子块构建生成逻辑。

### 4.2 接口

```python
class Generator(nn.Module):
    """参数生成网络基类。

    基类自动从 param_spec 派生便利属性，用户无需手动处理 param_spec 字典。

    Args:
        param_spec (dict): 目标参数规格，由 MappingLayer 自动传入。
            格式: {'weight': (C_out, C_in, kh, kw), 'bias': (C_out,)}
            当 bias=False 时，不含 'bias' 键。
        z_dim (int): 隐变量 z 的维度，必须显式声明。
        **kwargs: 用户自定义参数（如隐藏层大小等）。

    自动派生属性:
        self.w_shape  (tuple):  weight 目标形状，如 (20, 1, 5, 5)
        self.b_shape  (tuple | None): bias 目标形状，如 (20,) 或 None
        self.w_size   (int):   weight 总元素数，如 500
        self.b_size   (int):   bias 总元素数，如 20 或 0
    """

    def __init__(self, param_spec: dict, z_dim: int, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim))  # 可训练

        # 自动派生 — 用户直接使用，无需手动访问 param_spec
        self.w_shape = param_spec['weight']
        self.b_shape = param_spec.get('bias')       # bias=False 时为 None
        self.w_size = prod(self.w_shape)
        self.b_size = prod(self.b_shape) if self.b_shape else 0

    @abstractmethod
    def forward(self) -> tuple:
        """返回生成的参数张量。

        Returns:
            tuple: (weight, bias)
                - weight: 形状为 self.w_shape 的张量，或 1D flat
                - bias:   形状为 self.b_shape 的张量，或 1D flat（bias=False 时为 None）
                - 默认 shaped：张量已匹配目标形状，MappingLayer 直接使用
                - flat 模式：1D 张量，MappingLayer 的 _resolve 自动 reshape
        """
        raise NotImplementedError
```

### 4.3 关键设计

- **`self.z` 是唯一的可训练参数**（`nn.Parameter`，`requires_grad=True`）
- 基类自动派生 `w_shape/b_shape/w_size/b_size`，用户编写 generator 时直接用，无需手动访问 `param_spec`
- `forward()` 返回 tuple `(weight, bias)`，简洁直观
- **shaped 优先**：默认输出 shaped 张量；flat 模式作为可选支持（`_resolve` 自动处理）

### 4.4 示例

```python
class MyGen(mapping.Generator):
    def __init__(self, param_spec, z_dim=64, hidden_dim=128):
        super().__init__(param_spec, z_dim=z_dim)
        # 基类自动派生 self.w_shape, self.w_size 等，直接用
        self.body = nn.Sequential(
            mapping.generator.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            mapping.generator.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.w_head = nn.Linear(hidden_dim * 2, self.w_size)   # self.w_size = 500
        self.b_head = nn.Linear(hidden_dim * 2, self.b_size)   # self.b_size = 20

    def forward(self):
        h = self.body(self.z)
        return (
            self.w_head(h).reshape(self.w_shape),  # 形状: (20, 1, 5, 5)
            self.b_head(h).reshape(self.b_shape),  # 形状: (20,)
        )
```

---

## 5. Layer 3: MappingLayer (`mapping.Conv2d` / `mapping.Linear`)

### 5.1 定位

主干网络的层，init 签名严格对齐 torch，`param_spec` 自动推导。支持两种模式：

- **LWT 模式**：传入 `generator_cls`，层自行实例化 generator
- **SLVT 模式**：不传 generator 参数，作为纯形状层，由 Sequential 喂参数

### 5.2 基类

```python
class MappingLayer(nn.Module):
    """主干网络层基类。

    子类需实现:
        - _build_param_spec() -> dict: 返回 {'weight': shape, 'bias': shape}
        - _functional(x, params) -> Tensor: 用 params 执行函数式前向
    """

    def _resolve(self, t: Tensor, target_shape: tuple) -> Tensor:
        """解析张量形状：shaped 直通，flat 则 reshape。"""
        return t if t.shape == target_shape else t.reshape(target_shape)

    def forward(self, x) -> Tensor:
        """LWT 入口：调用自己的 generator → _functional。"""
        w, b = self.generator()
        return self._functional(x, w, b)

    def forward_with_params(self, x, w, b) -> Tensor:
        """SLVT 入口：接收外部参数 tuple → _functional。"""
        return self._functional(x, w, b)
```

### 5.3 `mapping.Conv2d`

```python
class Conv2d(MappingLayer):
    """2D 卷积映射层。

    Args:
        in_channels  (int): 输入通道数 C_in
        out_channels (int): 输出通道数 C_out
        kernel_size  (int | tuple): 卷积核尺寸 (kh, kw)
        stride       (int | tuple): 步长 (默认 1)
        padding      (int | tuple): 填充 (默认 0)
        dilation     (int | tuple): 膨胀 (默认 1)
        groups       (int): 分组卷积数 (默认 1)
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        **generator_kwargs: 透传给 generator 构造函数的参数（如 z_dim, hidden_dim 等）

    param_spec:
        weight: (C_out, C_in, kh, kw)
            总元素数 = C_out * C_in * kh * kw
        bias:   (C_out,)
            总元素数 = C_out  (仅 bias=True 时)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 generator_cls=None, **generator_kwargs):
        # 自动推导 kernel_size 为 (kh, kw)
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        # 自动构建 param_spec
        param_spec = {
            'weight': (out_channels, in_channels, kh, kw),
            # 总元素数 = out_channels * in_channels * kh * kw
        }
        if bias:
            param_spec['bias'] = (out_channels,)
            # 总元素数 = out_channels

        self.param_spec = param_spec
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        # 实例化 generator
        if generator_cls is not None:
            self.generator = generator_cls(param_spec, **generator_kwargs)

    def _functional(self, x, w, b):
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
```

### 5.4 `mapping.Linear`

```python
class Linear(MappingLayer):
    """线性映射层。

    Args:
        in_features  (int): 输入特征数 N_in
        out_features (int): 输出特征数 N_out
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        **generator_kwargs: 透传给 generator 构造函数的参数（如 z_dim, hidden_dim 等）

    param_spec:
        weight: (N_out, N_in)
            总元素数 = N_out * N_in
        bias:   (N_out,)
            总元素数 = N_out  (仅 bias=True 时)
    """

    def _functional(self, x, w, b):
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.linear(x, w, b)
```

### 5.5 互斥规则

- 传了 `generator_cls` → LWT 层
- 都没传 → 纯形状层（只能被 Sequential 喂参数）
- Sequential 中包含自带 generator 的层 → **构造时报错**

---

## 6. `mapping.Sequential` (SLVT 容器)

### 6.1 定位

SLVT 模式的容器，持有一个共享 generator，管理所有参数层的参数。weight 和 bias 沿两条独立的 flat 线分别切片。

### 6.2 接口

```python
class Sequential(nn.Module):
    """SLVT 模式的共享 generator 容器。

    Args:
        *layers: 纯形状 MappingLayer（不能自带 generator），可混装非参数层
        generator_cls: Generator 子类
        **generator_kwargs: 透传给 generator 构造函数的参数（如 z_dim, hidden_dim 等）
    """

    def __init__(self, *layers, generator_cls, **generator_kwargs):
        # 1. 验证互斥：不能包含自带 generator 的层
        for l in layers:
            if isinstance(l, MappingLayer) and hasattr(l, 'generator') and l.generator is not None:
                raise ValueError(
                    f"Sequential 中的层不能自带 generator，"
                    f"但 {l} 已配置了 generator。"
                )

        # 2. 收集所有参数层的 weight/bias 大小，算切片边界
        w_total, b_total = 0, 0
        self.w_bounds, self.b_bounds = [0], [0]

        for l in layers:
            if isinstance(l, MappingLayer):
                spec = l.param_spec
                w_total += prod(spec['weight'])
                self.w_bounds.append(w_total)
                if 'bias' in spec:
                    b_total += prod(spec['bias'])
                    self.b_bounds.append(b_total)
                else:
                    self.b_bounds.append(b_total)  # 无 bias 层不贡献长度
            else:
                # 非参数层不贡献切片
                self.w_bounds.append(w_total)
                self.b_bounds.append(b_total)

        # 3. 创建共享 generator
        full_spec = {
            'weight': (w_total,),
            'bias': (b_total,) if b_total > 0 else None,
        }
        self.generator = generator_cls(full_spec, **generator_kwargs)

    def forward(self, x):
        flat_w, flat_b = self.generator()  # tuple (flat_weight, flat_bias)
        param_idx = 0

        for l in self.layers:
            if isinstance(l, MappingLayer):
                ws, we = self.w_bounds[param_idx], self.w_bounds[param_idx + 1]
                bs, be = self.b_bounds[param_idx], self.b_bounds[param_idx + 1]

                w_slice = flat_w[ws:we]      # 一维 flat 切片，_resolve 会 reshape
                b_slice = flat_b[bs:be] if flat_b is not None and be > bs else None

                x = l.forward_with_params(x, w_slice, b_slice)
                param_idx += 1
            else:
                x = l(x)  # 非参数层直接串联

        return x
```

### 6.3 关键设计

- weight 和 bias 沿**两条独立 flat** 分别切片（不合并）
- 非参数层（`nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten` 等）直接串联，不参与切片
- 共享 generator 的 `param_spec` 是两条一维 flat：`{'weight': (total_w,), 'bias': (total_b,)}`，forward 返回 tuple `(flat_w, flat_b)`

---

## 7. Generator 辅助模块

### 7.1 `mapping.generator.LRDLayer`

低秩分解辅助模块，供用户在 generator 内部使用。用户将其放入 generator 的 forward 中，即可实现 LRD。

```python
class LRDLayer(nn.Module):
    """低秩分解辅助模块。

    Args:
        m (int): 原始 weight 的行数
        n (int): 原始 weight 的列数
        rank (int): 低秩分解的秩 r

    用法（在 generator.forward 中):
        lrd = LRDLayer(m=512, n=176, rank=10)  # 不需要放在 generator 的 __init__ 中
        flat_input = head(z)                     # generator 输出 flat 张量
        U = flat_input[:m*rank].reshape(m, rank)
        V = flat_input[m*rank:m*rank+n*rank].reshape(n, rank)
        weight = U @ V.T                        # (m, n) = (512, 176)
        return weight, bias                     # tuple (weight, bias)
    """
```

---

## 8. 使用示例

### 8.1 LWT 模式（逐层，默认）

```python
class MyLWTNet(nn.Module):
    def __init__(self):
        self.conv1 = mapping.Conv2d(1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)
        self.conv2 = mapping.Conv2d(20, 32, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)
        self.fc1   = mapping.Linear(512, 176, generator_cls=MyGen, z_dim=64, hidden_dim=128)
        self.fc2   = mapping.Linear(176, 10, generator_cls=MyGen, z_dim=64, hidden_dim=128)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)
```

### 8.2 SLVT 模式（共享 generator）

```python
net = mapping.Sequential(
    mapping.Conv2d(1, 20, 5),      # 纯形状层
    nn.ReLU(),
    nn.MaxPool2d(2),
    mapping.Conv2d(20, 32, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(1),
    mapping.Linear(512, 176),
    nn.ReLU(),
    mapping.Linear(176, 10),
    generator_cls=MyGen,
    z_dim=64, hidden_dim=256,      # 直接透传给 generator
)
y = net(x)
```

### 8.3 基线（完全独立，纯手写 torch）

```python
class BaselineCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(512, 176)
        self.fc2 = nn.Linear(176, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))
```

---

## 9. 模块结构

```
mapping/                          # 全新框架包
    __init__.py                   # 导出: Generator, Conv2d, Linear, Sequential, LRDLayer
    base.py                       # Generator 基类, MappingLayer 基类
    layers.py                     # Conv2d, Linear 等具体层
    sequential.py                 # Sequential 容器
    generator/                    # Generator 子块包
        __init__.py               # 导出: Linear, Conv1d, Conv2d, LRDLayer
        linear.py                 # generator.Linear
        conv.py                   # generator.Conv1d, generator.Conv2d
        lrd.py                    # LRDLayer 辅助模块
```

---

## 10. 不在范围内

以下内容明确不在框架中实现，由用户自行负责：

- **损失函数**：L_task, L_stab, L_smooth, L_align 等
- **z 采样**：高斯采样、噪声注入等
- **训练循环**：Trainer、优化器、scheduler
- **Checkpoint**：保存/恢复逻辑
- **旧代码兼容**：现有 `mapping_network/` 的 `target_nets/`, `trainer/`, `mapping/loss.py` 等
- **基线训练**：纯手写 torch，与 mapping 框架无关

---

## 11. 待定 / 未来扩展

- 其他层类型：`Conv1d`, `ConvTranspose2d`, `BatchNorm` 等
- Generator 辅助模块：残差连接块、Transformer 块等
- `generator_instance` 传入方式：允许用户传入已实例化的 generator（支持跨层共享 generator 实例的特殊场景）
- 多 GPU / 分布式支持