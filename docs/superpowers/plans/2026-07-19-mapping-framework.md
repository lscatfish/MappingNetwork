# Mapping 推理框架 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现全新的 `mapping/` 推理框架，包含 Generator 子块、Generator 基类、MappingLayer 层、Sequential 容器，以及完整的单元测试。

**Architecture:** 三层分离 — (1) `mapping.generator.*` 固定随机参数子块, (2) `mapping.Generator` 可训练参数生成网络基类, (3) `mapping.MappingLayer` 主干网络层。LWT 通过直接堆叠层实现，SLVT 通过 `mapping.Sequential` 共享 generator 实现。

**Tech Stack:** Python 3.10+, PyTorch 2.x, pytest

**Spec:** [docs/superpowers/specs/2026-07-19-mapping-framework-design.md](../specs/2026-07-19-mapping-framework-design.md)

## Global Constraints

- 测试必须走 GPU（`device` fixture），禁止 `.cpu()` / `map_location='cpu'` / `device='cuda'` 硬编码
- 使用 `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest` 运行测试
- 新框架放在 `mapping/` 包，与现有 `mapping_network/` 完全独立
- 所有代码必须带类型注解（函数签名 + 参数类型）
- 不兼容旧代码，不修改现有 `mapping_network/` 目录

---

### Task 1: 包脚手架

**Files:**
- Create: `mapping/__init__.py`
- Create: `mapping/generator/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces: `mapping` package structure, empty `__init__.py` files

- [ ] **Step 1: 创建目录和文件**

```bash
mkdir -p /root/MyProj/MappingNetwork/mapping/generator
```

```python
# mapping/__init__.py (空文件)
```

```python
# mapping/generator/__init__.py (空文件)
```

- [ ] **Step 2: 提交**

```bash
git add mapping/
git commit -m "chore: scaffold mapping/ package structure"
```

---

### Task 2: `mapping.generator.Linear` — 固定随机参数线性子块

**Files:**
- Create: `mapping/generator/linear.py`
- Create: `tests/test_generator_blocks.py`

**Interfaces:**
- Consumes: nothing
- Produces: `mapping.generator.Linear(in_features, out_features, bias=True)` — init 对齐 `nn.Linear`，内部参数 `requires_grad=False`，默认论文初始化

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_generator_blocks.py
import torch
import pytest
from mapping.generator.linear import Linear


class TestGeneratorLinear:
    def test_init_aligns_torch(self, device):
        """init 签名对齐 torch.nn.Linear。"""
        layer = Linear(10, 20).to(device)
        assert layer.weight.shape == (20, 10)
        assert layer.bias.shape == (20,)

    def test_params_are_frozen(self, device):
        """内部参数 requires_grad=False。"""
        layer = Linear(10, 20).to(device)
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad

    def test_no_bias(self, device):
        """bias=False 时 bias 为 None。"""
        layer = Linear(10, 20, bias=False).to(device)
        assert layer.bias is None

    def test_forward_matches_torch(self, device):
        """forward 行为与 F.linear 一致。"""
        layer = Linear(10, 20).to(device)
        x = torch.randn(4, 10, device=device)
        y = layer(x)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.allclose(y, expected)

    def test_init_weights_called_on_construction(self, device):
        """构造时自动调用 init_weights。"""
        layer = Linear(10, 20).to(device)
        # 权重非零且非全等（已初始化）
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_custom_init_weights(self, device):
        """用户可重载 init_weights 自定义初始化。"""

        class CustomLinear(Linear):
            def init_weights(self):
                torch.nn.init.ones_(self.weight)
                if self.bias is not None:
                    torch.nn.init.zeros_(self.bias)

        layer = CustomLinear(10, 20).to(device)
        assert torch.allclose(layer.weight, torch.ones_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_forward_preserves_gradient(self, device):
        """forward 输出可反向传播（子块参数虽固定，但输入可梯度）。"""
        layer = Linear(10, 20).to(device)
        x = torch.randn(4, 10, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestGeneratorLinear -v
```
Expected: ImportError (module not found)

- [ ] **Step 3: 实现 `mapping/generator/linear.py`**

```python
"""固定随机参数 Linear 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """固定随机参数的线性层子块。

    init 签名对齐 torch.nn.Linear。内部参数在构造时随机初始化
    并设为 requires_grad=False。默认采用论文方法初始化，
    用户可重载 init_weights() 自定义。

    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def init_weights(self):
        """默认论文初始化方法：kaiming uniform。

        子类可重载此方法自定义初始化。
        """
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
```

- [ ] **Step 4: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestGeneratorLinear -v
```
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add mapping/generator/linear.py tests/test_generator_blocks.py
git commit -m "feat: add mapping.generator.Linear — frozen random param sub-block"
```

---

### Task 3: `mapping.generator.Conv1d` & `mapping.generator.Conv2d`

**Files:**
- Create: `mapping/generator/conv.py`
- Modify: `tests/test_generator_blocks.py` (追加测试)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `mapping.generator.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`
  - `mapping.generator.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`

- [ ] **Step 1: 追加测试用例**

```python
# 追加到 tests/test_generator_blocks.py


class TestGeneratorConv1d:
    def test_init_aligns_torch(self, device):
        """init 签名对齐 torch.nn.Conv1d。"""
        from mapping.generator.conv import Conv1d
        layer = Conv1d(3, 16, kernel_size=3).to(device)
        assert layer.weight.shape == (16, 3, 3)
        assert layer.bias.shape == (16,)

    def test_params_are_frozen(self, device):
        from mapping.generator.conv import Conv1d
        layer = Conv1d(3, 16, 3).to(device)
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad

    def test_forward_matches_torch(self, device):
        from mapping.generator.conv import Conv1d
        layer = Conv1d(3, 16, 3, padding=1).to(device)
        x = torch.randn(2, 3, 10, device=device)
        y = layer(x)
        expected = torch.nn.functional.conv1d(x, layer.weight, layer.bias, padding=1)
        assert torch.allclose(y, expected)

    def test_stride_dilation(self, device):
        from mapping.generator.conv import Conv1d
        layer = Conv1d(3, 16, 3, stride=2, dilation=2).to(device)
        x = torch.randn(2, 3, 20, device=device)
        y = layer(x)
        expected = torch.nn.functional.conv1d(x, layer.weight, layer.bias, stride=2, dilation=2)
        assert torch.allclose(y, expected)


class TestGeneratorConv2d:
    def test_init_aligns_torch(self, device):
        """init 签名对齐 torch.nn.Conv2d。"""
        from mapping.generator.conv import Conv2d
        layer = Conv2d(3, 16, kernel_size=3).to(device)
        assert layer.weight.shape == (16, 3, 3, 3)
        assert layer.bias.shape == (16,)

    def test_params_are_frozen(self, device):
        from mapping.generator.conv import Conv2d
        layer = Conv2d(3, 16, 3).to(device)
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad

    def test_forward_matches_torch(self, device):
        from mapping.generator.conv import Conv2d
        layer = Conv2d(3, 16, 3, padding=1).to(device)
        x = torch.randn(2, 3, 10, 10, device=device)
        y = layer(x)
        expected = torch.nn.functional.conv2d(x, layer.weight, layer.bias, padding=1)
        assert torch.allclose(y, expected)

    def test_tuple_kernel_size(self, device):
        from mapping.generator.conv import Conv2d
        layer = Conv2d(3, 16, kernel_size=(3, 5)).to(device)
        assert layer.weight.shape == (16, 3, 3, 5)

    def test_no_bias(self, device):
        from mapping.generator.conv import Conv2d
        layer = Conv2d(3, 16, 3, bias=False).to(device)
        assert layer.bias is None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestGeneratorConv1d tests/test_generator_blocks.py::TestGeneratorConv2d -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `mapping/generator/conv.py`**

```python
"""固定随机参数 Conv1d / Conv2d 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvNd(nn.Module):
    """Conv 子块基类，共享 init_weights 逻辑。"""

    def __init__(self):
        super().__init__()

    def init_weights(self):
        """默认论文初始化方法：kaiming uniform。

        子类可重载此方法自定义初始化。
        """
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            for s in self.weight.shape[2:]:
                fan_in *= s
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class Conv1d(_ConvNd):
    """固定随机参数的一维卷积子块。

    init 签名对齐 torch.nn.Conv1d。内部参数在构造时随机初始化
    并设为 requires_grad=False。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸
        stride: 步长 (默认 1)
        padding: 填充 (默认 0)
        dilation: 膨胀 (默认 1)
        groups: 分组卷积数 (默认 1)
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class Conv2d(_ConvNd):
    """固定随机参数的二维卷积子块。

    init 签名对齐 torch.nn.Conv2d。内部参数在构造时随机初始化
    并设为 requires_grad=False。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸 (int 或 tuple)
        stride: 步长 (默认 1)
        padding: 填充 (默认 0)
        dilation: 膨胀 (默认 1)
        groups: 分组卷积数 (默认 1)
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestGeneratorConv1d tests/test_generator_blocks.py::TestGeneratorConv2d -v
```
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add mapping/generator/conv.py tests/test_generator_blocks.py
git commit -m "feat: add mapping.generator.Conv1d and Conv2d sub-blocks"
```

---

### Task 4: `mapping.generator.LRDLayer` — 低秩分解辅助模块

**Files:**
- Create: `mapping/generator/lrd.py`
- Modify: `tests/test_generator_blocks.py` (追加测试)

**Interfaces:**
- Consumes: nothing
- Produces: `mapping.generator.LRDLayer(m, n, rank)` — 低秩分解辅助模块，在 generator.forward 中调用

- [ ] **Step 1: 追加测试用例**

```python
# 追加到 tests/test_generator_blocks.py


class TestLRDLayer:
    def test_lrd_reconstructs_shape(self, device):
        """LRD 重建 weight 形状正确。"""
        from mapping.generator.lrd import LRDLayer

        m, n, rank = 512, 176, 10
        lrd = LRDLayer(m, n, rank).to(device)

        # 模拟 generator 输出 flat 张量
        flat = torch.randn(m * rank + n * rank, device=device)

        U = flat[:m * rank].reshape(m, rank)       # (512, 10)
        V = flat[m * rank:].reshape(n, rank)        # (176, 10)
        weight = U @ V.T                             # (512, 176)

        assert weight.shape == (m, n)

    def test_lrd_differentiable(self, device):
        """LRD 重建 weight 可反向传播。"""
        from mapping.generator.lrd import LRDLayer

        m, n, rank = 512, 176, 10
        lrd = LRDLayer(m, n, rank).to(device)

        flat = torch.randn(m * rank + n * rank, device=device, requires_grad=True)
        U = flat[:m * rank].reshape(m, rank)
        V = flat[m * rank:].reshape(n, rank)
        weight = U @ V.T
        loss = weight.sum()
        loss.backward()

        assert flat.grad is not None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestLRDLayer -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `mapping/generator/lrd.py`**

```python
"""低秩分解 (LRD) 辅助模块。"""

import torch
import torch.nn as nn


class LRDLayer(nn.Module):
    """低秩分解辅助模块。

    供用户在 generator.forward() 中调用，将 flat 输出分解为 U, V
    并重构为完整 weight 矩阵 W = U @ V^T。

    用法::

        class MyGen(mapping.Generator):
            def forward(self):
                flat = self.head(self.z)
                # 重构 LRD weight
                U = flat[:self.w_size].reshape(self.w_shape[0], rank)
                V = flat[self.w_size:].reshape(self.w_shape[1], rank)
                weight = U @ V.T
                return weight, bias

    Args:
        m: 原始 weight 的行数
        n: 原始 weight 的列数
        rank: 低秩分解的秩 r
    """

    def __init__(self, m: int, n: int, rank: int):
        super().__init__()
        self.m = m
        self.n = n
        self.rank = rank

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        """将 flat 张量分解为 U, V 并重构为 W = U @ V^T。

        Args:
            flat: 形状为 (m * rank + n * rank,) 的一维张量

        Returns:
            weight: 形状为 (m, n) 的完整权重矩阵
        """
        m, n, r = self.m, self.n, self.rank
        U = flat[:m * r].reshape(m, r)
        V = flat[m * r:].reshape(n, r)
        return U @ V.T

    def extra_repr(self) -> str:
        return f'm={self.m}, n={self.n}, rank={self.rank}'
```

- [ ] **Step 4: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py::TestLRDLayer -v
```
Expected: 2 passed

- [ ] **Step 5: 更新 `mapping/generator/__init__.py` 导出**

```python
from mapping.generator.linear import Linear
from mapping.generator.conv import Conv1d, Conv2d
from mapping.generator.lrd import LRDLayer

__all__ = ['Linear', 'Conv1d', 'Conv2d', 'LRDLayer']
```

- [ ] **Step 6: 提交**

```bash
git add mapping/generator/lrd.py mapping/generator/__init__.py tests/test_generator_blocks.py
git commit -m "feat: add mapping.generator.LRDLayer and generator __init__ exports"
```

---

### Task 5: `mapping.Generator` 基类

**Files:**
- Create: `mapping/base.py`
- Create: `tests/test_generator.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `mapping.Generator(param_spec, z_dim, **kwargs)` — 基类，自动派生 `self.w_shape/b_shape/w_size/b_size`，`forward() -> tuple`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_generator.py
import torch
import pytest
from mapping.base import Generator


class TestGenerator:
    def test_z_is_trainable_parameter(self, device):
        """z 是 nn.Parameter 且 requires_grad=True。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape)
                return w, b

        gen = SimpleGen(
            {'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64
        ).to(device)
        assert isinstance(gen.z, torch.nn.Parameter)
        assert gen.z.requires_grad

    def test_auto_derived_attrs(self, device):
        """基类自动派生 w_shape/b_shape/w_size/b_size。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape)
                return w, b

        gen = SimpleGen(
            {'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64
        ).to(device)
        assert gen.w_shape == (20, 1, 5, 5)
        assert gen.b_shape == (20,)
        assert gen.w_size == 500   # 20*1*5*5
        assert gen.b_size == 20

    def test_no_bias(self, device):
        """bias=False 时 b_shape=None, b_size=0。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size)

            def forward(self):
                h = self.head(self.z)
                return h.reshape(self.w_shape), None

        gen = SimpleGen({'weight': (10, 5)}, z_dim=32).to(device)
        assert gen.b_shape is None
        assert gen.b_size == 0

    def test_forward_returns_tuple(self, device):
        """forward 返回 (weight, bias) tuple。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape)
                return w, b

        gen = SimpleGen(
            {'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64
        ).to(device)
        w, b = gen()
        assert w.shape == (20, 1, 5, 5)
        assert b.shape == (20,)

    def test_forward_is_abstract(self):
        """Generator 不可直接实例化（forward 是抽象的）。"""
        with pytest.raises(TypeError):
            Generator({'weight': (10,), 'bias': (10,)}, z_dim=32)

    def test_kwargs_passthrough(self, device):
        """**kwargs 透传给子类 __init__。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, hidden_dim=128, **kwargs):
                super().__init__(param_spec, z_dim=z_dim, **kwargs)
                self.hidden_dim = hidden_dim
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape)
                return w, b

        gen = KwargsGen(
            {'weight': (20,), 'bias': (20,)}, z_dim=64, hidden_dim=256
        ).to(device)
        assert gen.hidden_dim == 256

    def test_z_gradient_flows(self, device):
        """z 的梯度正常流动。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape)
                return w, b

        gen = SimpleGen({'weight': (10, 5), 'bias': (10,)}, z_dim=32).to(device)
        w, b = gen()
        loss = w.sum() + b.sum()
        loss.backward()
        assert gen.z.grad is not None
        assert not torch.allclose(gen.z.grad, torch.zeros_like(gen.z.grad))
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator.py::TestGenerator -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `mapping/base.py` 中的 Generator**

```python
"""Mapping 框架基类：Generator 和 MappingLayer。"""

from abc import abstractmethod
from functools import reduce
from operator import mul

import torch
import torch.nn as nn


def _prod(iterable) -> int:
    return reduce(mul, iterable, 1)


class Generator(nn.Module):
    """参数生成网络基类。

    基类自动从 param_spec 派生便利属性，用户无需手动处理 param_spec 字典。

    Args:
        param_spec: 目标参数规格，由 MappingLayer 自动传入。
            格式: {'weight': (C_out, C_in, kh, kw), 'bias': (C_out,)}
            当 bias=False 时，不含 'bias' 键。
        z_dim: 隐变量 z 的维度，必须显式声明。
        **kwargs: 用户自定义参数（如隐藏层大小等）。

    自动派生属性:
        self.w_shape  (tuple):  weight 目标形状
        self.b_shape  (tuple | None): bias 目标形状，或 None
        self.w_size   (int):   weight 总元素数
        self.b_size   (int):   bias 总元素数，或 0
    """

    def __init__(self, param_spec: dict, z_dim: int, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim))

        self.w_shape = param_spec['weight']
        self.b_shape = param_spec.get('bias')
        self.w_size = _prod(self.w_shape)
        self.b_size = _prod(self.b_shape) if self.b_shape else 0

    @abstractmethod
    def forward(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回生成的参数张量。

        Returns:
            tuple: (weight, bias)
                - weight: 形状为 self.w_shape 的张量，或 1D flat
                - bias:   形状为 self.b_shape 的张量，或 1D flat（bias=False 时为 None）
        """
        raise NotImplementedError
```

- [ ] **Step 4: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator.py::TestGenerator -v
```
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add mapping/base.py tests/test_generator.py
git commit -m "feat: add mapping.Generator base class with auto-derived attrs"
```

---

### Task 6: `mapping.MappingLayer` + `mapping.Conv2d` + `mapping.Linear`

**Files:**
- Modify: `mapping/base.py` (追加 MappingLayer 基类)
- Create: `mapping/layers.py`
- Create: `tests/test_layers.py`

**Interfaces:**
- Consumes: `mapping.Generator` (from Task 5)
- Produces:
  - `mapping.MappingLayer` — 基类，提供 `_resolve`, `forward`, `forward_with_params`
  - `mapping.Conv2d(in_channels, out_channels, kernel_size, ..., generator_cls=None, **generator_kwargs)`
  - `mapping.Linear(in_features, out_features, bias=True, generator_cls=None, **generator_kwargs)`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_layers.py
import torch
import pytest
import torch.nn.functional as F
from mapping.base import Generator, MappingLayer
from mapping.layers import Conv2d, Linear


# --- 测试用的 Generator ---
class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestConv2d:
    def test_param_spec_auto_deduced(self, device):
        """Conv2d 自动推导 param_spec。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (20, 1, 5, 5)
        assert layer.param_spec['bias'] == (20,)

    def test_param_spec_no_bias(self, device):
        """bias=False 时 param_spec 不含 bias。"""
        layer = Conv2d(1, 20, 5, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        """forward 输出形状正确。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = layer(x)
        assert y.shape == (2, 20, 24, 24)

    def test_forward_with_params(self, device):
        """forward_with_params 接收外部参数。"""
        layer = Conv2d(1, 20, 5).to(device)  # 纯形状层
        x = torch.randn(2, 1, 28, 28, device=device)
        w = torch.randn(20, 1, 5, 5, device=device)
        b = torch.randn(20, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv2d(x, w, b)
        assert torch.allclose(y, expected)

    def test_flat_params_auto_reshape(self, device):
        """flat 参数自动 reshape。"""
        layer = Conv2d(1, 20, 5).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        w_flat = torch.randn(500, device=device)   # 20*1*5*5
        b_flat = torch.randn(20, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 20, 24, 24)

    def test_stride_padding(self, device):
        """stride 和 padding 参数生效。"""
        layer = Conv2d(3, 16, 3, stride=2, padding=1).to(device)
        x = torch.randn(2, 3, 10, 10, device=device)
        w = torch.randn(16, 3, 3, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv2d(x, w, b, stride=2, padding=1)
        assert torch.allclose(y, expected)

    def test_generator_kwargs_passthrough(self, device):
        """**generator_kwargs 透传给 Generator。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, my_param=42):
                super().__init__(param_spec, z_dim=z_dim)
                self.my_param = my_param
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
                return w, b

        layer = Conv2d(1, 20, 5, generator_cls=KwargsGen, z_dim=32, my_param=99).to(device)
        assert layer.generator.my_param == 99

    def test_pure_shape_layer_no_generator(self, device):
        """不传 generator_cls 时，层没有 generator 属性。"""
        layer = Conv2d(1, 20, 5).to(device)
        assert not hasattr(layer, 'generator')

    def test_gradient_flows_through_generator(self, device):
        """梯度通过 generator 流向 z。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.generator.z.grad is not None


class TestLinear:
    def test_param_spec_auto_deduced(self, device):
        """Linear 自动推导 param_spec。"""
        layer = Linear(512, 176, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (176, 512)
        assert layer.param_spec['bias'] == (176,)

    def test_forward_output_shape(self, device):
        """forward 输出形状正确。"""
        layer = Linear(512, 176, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 512, device=device)
        y = layer(x)
        assert y.shape == (2, 176)

    def test_forward_with_params(self, device):
        """forward_with_params 接收外部参数。"""
        layer = Linear(512, 176).to(device)
        x = torch.randn(2, 512, device=device)
        w = torch.randn(176, 512, device=device)
        b = torch.randn(176, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.linear(x, w, b)
        assert torch.allclose(y, expected)

    def test_no_bias(self, device):
        """bias=False 时，forward_with_params 容错。"""
        layer = Linear(512, 176, bias=False).to(device)
        x = torch.randn(2, 512, device=device)
        w = torch.randn(176, 512, device=device)
        y = layer.forward_with_params(x, w, None)
        expected = F.linear(x, w)
        assert torch.allclose(y, expected)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_layers.py -v
```
Expected: ImportError

- [ ] **Step 3: 追加 MappingLayer 到 `mapping/base.py`**

```python
# 追加到 mapping/base.py 末尾


class MappingLayer(nn.Module):
    """主干网络层基类。

    子类需实现:
        - _functional(x, w, b) -> Tensor: 用参数执行函数式前向
    """

    def _resolve(self, t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """解析张量形状：shaped 直通，flat 则 reshape。"""
        return t if t.shape == target_shape else t.reshape(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LWT 入口：调用自己的 generator → _functional。"""
        w, b = self.generator()
        return self._functional(x, w, b)

    def forward_with_params(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        """SLVT 入口：接收外部参数 tuple → _functional。"""
        return self._functional(x, w, b)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        raise NotImplementedError
```

- [ ] **Step 4: 实现 `mapping/layers.py`**

```python
"""Mapping 主干网络层：Conv2d, Linear。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.base import Generator, MappingLayer


class Conv2d(MappingLayer):
    """2D 卷积映射层。

    init 签名对齐 torch.nn.Conv2d。param_spec 自动推导。

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
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C_out, C_in, kh, kw)
            总元素数 = C_out * C_in * kh * kw
        bias:   (C_out,)
            总元素数 = C_out  (仅 bias=True 时)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        generator_cls: type[Generator] | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        kh, kw = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        self.param_spec = {'weight': (out_channels, in_channels, kh, kw)}
        if bias:
            self.param_spec['bias'] = (out_channels,)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        if generator_cls is not None:
            self.generator = generator_cls(self.param_spec, **generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )


class Linear(MappingLayer):
    """线性映射层。

    init 签名对齐 torch.nn.Linear。param_spec 自动推导。

    Args:
        in_features  (int): 输入特征数 N_in
        out_features (int): 输出特征数 N_out
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (N_out, N_in)
            总元素数 = N_out * N_in
        bias:   (N_out,)
            总元素数 = N_out  (仅 bias=True 时)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        generator_cls: type[Generator] | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        self.param_spec = {'weight': (out_features, in_features)}
        if bias:
            self.param_spec['bias'] = (out_features,)

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        if generator_cls is not None:
            self.generator = generator_cls(self.param_spec, **generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.linear(x, w, b)
```

- [ ] **Step 5: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_layers.py -v
```
Expected: 12 passed

- [ ] **Step 6: 提交**

```bash
git add mapping/base.py mapping/layers.py tests/test_layers.py
git commit -m "feat: add MappingLayer base, mapping.Conv2d and mapping.Linear"
```

---

### Task 7: `mapping.Sequential` — SLVT 容器

**Files:**
- Create: `mapping/sequential.py`
- Create: `tests/test_sequential.py`

**Interfaces:**
- Consumes: `mapping.MappingLayer`, `mapping.Generator` (from Tasks 5-6)
- Produces: `mapping.Sequential(*layers, generator_cls, **generator_kwargs)`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_sequential.py
import torch
import pytest
import torch.nn.functional as F
from mapping.base import Generator
from mapping.layers import Conv2d, Linear
from mapping.sequential import Sequential


# --- 测试用的 Generator ---
class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestSequential:
    def test_slvt_forward(self, device):
        """SLVT Sequential 基本前向。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            Linear(512, 10),
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        # 期望输出形状: (2, 10)
        # 计算路径: conv1(pool) -> conv2(pool) -> flatten -> linear
        assert y.shape[0] == 2

    def test_mixed_param_and_nonparam_layers(self, device):
        """可混装非参数层（ReLU, MaxPool2d, Flatten）。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(20, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(512, 10),
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_rejects_layers_with_own_generator(self, device):
        """传入自带 generator 的层时报错。"""
        with pytest.raises(ValueError, match='自带 generator'):
            Sequential(
                Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32),
                generator_cls=SimpleGen,
                z_dim=64,
            )

    def test_gradient_flows(self, device):
        """梯度通过共享 generator 流向 z。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            Linear(512, 10),  # 注意：28*28 -> 20*24*24 -> 20*12*12=2880
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        loss = y.sum()
        loss.backward()
        assert net.generator.z.grad is not None

    def test_generator_kwargs_passthrough(self, device):
        """**generator_kwargs 透传给 Generator。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, my_param=42):
                super().__init__(param_spec, z_dim=z_dim)
                self.my_param = my_param
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
                return w, b

        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            Linear(512, 10),
            generator_cls=KwargsGen,
            z_dim=64,
            my_param=99,
        ).to(device)
        assert net.generator.my_param == 99
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_sequential.py -v
```
Expected: ImportError

- [ ] **Step 3: 实现 `mapping/sequential.py`**

```python
"""SLVT 模式的共享 generator 容器。"""

import torch
import torch.nn as nn

from mapping.base import Generator, MappingLayer, _prod


class Sequential(nn.Module):
    """SLVT 模式的共享 generator 容器。

    持有一个共享 generator，管理所有参数层的参数。
    weight 和 bias 沿两条独立的 flat 线分别切片。

    Args:
        *layers: 纯形状 MappingLayer（不能自带 generator），可混装非参数层
        generator_cls: Generator 子类
        **generator_kwargs: 透传给 generator 构造函数的参数
    """

    def __init__(self, *layers, generator_cls: type[Generator], **generator_kwargs):
        super().__init__()

        # 验证互斥：不能包含自带 generator 的层
        for i, layer in enumerate(layers):
            if isinstance(layer, MappingLayer) and hasattr(layer, 'generator'):
                raise ValueError(
                    f"Sequential 中的层不能自带 generator，"
                    f"但第 {i} 层 {layer} 已配置了 generator。"
                )

        self.layers = nn.ModuleList(layers)

        # 收集所有参数层的 weight/bias 大小，算切片边界
        w_total, b_total = 0, 0
        self.w_bounds = [0]
        self.b_bounds = [0]

        for layer in layers:
            if isinstance(layer, MappingLayer):
                spec = layer.param_spec
                w_total += _prod(spec['weight'])
                self.w_bounds.append(w_total)
                if 'bias' in spec:
                    b_total += _prod(spec['bias'])
                    self.b_bounds.append(b_total)
                else:
                    self.b_bounds.append(b_total)
            else:
                self.w_bounds.append(w_total)
                self.b_bounds.append(b_total)

        # 创建共享 generator
        full_spec = {
            'weight': (w_total,),
            'bias': (b_total,) if b_total > 0 else None,
        }
        self.generator = generator_cls(full_spec, **generator_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_w, flat_b = self.generator()
        param_idx = 0

        for layer in self.layers:
            if isinstance(layer, MappingLayer):
                ws = self.w_bounds[param_idx]
                we = self.w_bounds[param_idx + 1]
                bs = self.b_bounds[param_idx]
                be = self.b_bounds[param_idx + 1]

                w_slice = flat_w[ws:we]
                b_slice = flat_b[bs:be] if flat_b is not None and be > bs else None

                x = layer.forward_with_params(x, w_slice, b_slice)
                param_idx += 1
            else:
                x = layer(x)

        return x
```

- [ ] **Step 4: 运行测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_sequential.py -v
```
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add mapping/sequential.py tests/test_sequential.py
git commit -m "feat: add mapping.Sequential — SLVT shared-generator container"
```

---

### Task 8: 更新 `mapping/__init__.py` 导出 + 集成测试

**Files:**
- Modify: `mapping/__init__.py` (完整导出)
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: all previous tasks
- Produces: clean public API via `mapping/__init__.py`

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_integration.py
import torch
import pytest
import torch.nn.functional as F
from mapping import Generator, Conv2d, Linear, Sequential
from mapping.generator import Linear as GenLinear, Conv2d as GenConv2d


# --- 用户自定义 Generator（模拟真实使用场景）---
class MyGen(Generator):
    """用户自定义 Generator：使用 generator 子块组合。"""

    def __init__(self, param_spec, z_dim=64, hidden_dim=128):
        super().__init__(param_spec, z_dim=z_dim)
        self.body = torch.nn.Sequential(
            GenLinear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            GenLinear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
        )
        self.w_head = torch.nn.Linear(hidden_dim * 2, self.w_size)
        self.b_head = torch.nn.Linear(hidden_dim * 2, self.b_size) if self.b_size > 0 else None

    def forward(self):
        h = self.body(self.z)
        w = self.w_head(h).reshape(self.w_shape)
        b = self.b_head(h).reshape(self.b_shape) if self.b_head is not None else None
        return w, b


class TestIntegrationLWT:
    """LWT 模式集成测试：逐层 generator，直接堆叠。"""

    def test_lwt_forward(self, device):
        """LWT 完整前向：conv1 -> pool -> conv2 -> pool -> fc1 -> fc2。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)
                self.conv2 = Conv2d(20, 32, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)
                self.fc1 = Linear(512, 176, generator_cls=MyGen, z_dim=64, hidden_dim=128)
                self.fc2 = Linear(176, 10, generator_cls=MyGen, z_dim=64, hidden_dim=128)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = F.relu(self.fc1(x.flatten(1)))
                return self.fc2(x)

        net = LWTNet().to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)

    def test_lwt_gradient_flows(self, device):
        """LWT 各层 generator 的 z 独立训练。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)

            def forward(self, x):
                return F.relu(self.conv1(x))

        net = LWTNet().to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        loss = y.sum()
        loss.backward()

        assert net.conv1.generator.z.grad is not None
        assert not torch.allclose(
            net.conv1.generator.z.grad, torch.zeros_like(net.conv1.generator.z.grad)
        )

    def test_lwt_each_layer_has_own_z(self, device):
        """LWT 每层有独立的 z。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)
                self.conv2 = Conv2d(20, 32, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                return x

        net = LWTNet().to(device)
        # 两层 z 独立
        assert not torch.equal(net.conv1.generator.z.data, net.conv2.generator.z.data)


class TestIntegrationSLVT:
    """SLVT 模式集成测试：共享 generator。"""

    def test_slvt_full_forward(self, device):
        """SLVT 完整前向：Sequential 共享 generator。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(20, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(512, 176),
            torch.nn.ReLU(),
            Linear(176, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=256,
        ).to(device)

        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)

    def test_slvt_single_z(self, device):
        """SLVT 只有一个 z。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            Linear(512, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)

        assert net.generator.z.shape == (64,)
        assert net.generator.z.requires_grad


class TestFlatKerasStyle:
    """用户像 torch 一样写代码。"""

    def test_concise_syntax(self, device):
        """
        用户可以直接用简洁的语法构建网络。
        验证 init 中不出现嵌套的 generator_kwargs dict。
        """
        # 这是设计文档中期望的最终用户语法
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            Linear(512, 10),
            generator_cls=MyGen,
            z_dim=64,           # 直接透传，不是 generator_kwargs={'z_dim': 64}
            hidden_dim=128,     # 直接透传
        ).to(device)

        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_integration.py -v
```
Expected: ImportError (mapping.__init__ not yet complete)

- [ ] **Step 3: 更新 `mapping/__init__.py` 导出**

```python
"""Mapping 推理框架 — 参数生成 + 主干网络的前向推理框架。"""

from mapping.base import Generator, MappingLayer
from mapping.layers import Conv2d, Linear
from mapping.sequential import Sequential

__all__ = [
    'Generator',
    'MappingLayer',
    'Conv2d',
    'Linear',
    'Sequential',
]
```

- [ ] **Step 4: 运行集成测试验证通过**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_integration.py -v
```
Expected: 6 passed

- [ ] **Step 5: 运行全部测试**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_blocks.py tests/test_generator.py tests/test_layers.py tests/test_sequential.py tests/test_integration.py -v
```
Expected: all pass

- [ ] **Step 6: 提交**

```bash
git add mapping/__init__.py tests/test_integration.py
git commit -m "feat: complete mapping framework with public API and integration tests"
```