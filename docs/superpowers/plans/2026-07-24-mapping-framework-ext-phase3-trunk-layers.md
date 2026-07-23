# Phase 3: 主干层扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 mapping 框架新增 Conv1d、ConvTranspose2d、BatchNorm1d/2d 主干层和 trunk 级 ResBlock 容器，使框架支持更丰富的网络结构。

**Architecture:** 新增层均继承 `MappingLayer`，遵循现有 Conv2d/Linear 的模式（param_spec 自动推导 + 函数式前向 + generator_cls/generator_instance 双模式）。ResBlock 是特殊的 MappingLayer 容器：LWT 模式下内部各层自带 generator；SLVT 模式下作为纯形状层，param_spec 为聚合 flat，forward_with_params 内部二次切片。

**Tech Stack:** PyTorch (F.conv1d, F.conv_transpose2d, F.batch_norm), pytest, uv

## Global Constraints

- Python 3.13，所有命令用 `uv run python3` 或 `.venv/bin/python`
- 测试使用 `device` fixture（session 级，自动检测 CUDA），禁止硬编码 device
- 禁止修改 `mapping_network/` 旧包内部逻辑
- Ruff: line-length=100, 单引号, E/F/I 规则, 忽略 E501
- 每个 Task 结束后运行全量测试确认无回归
- commit 消息前缀: `feat:` / `fix:` / `test:`

---

## File Structure

| 文件 | 职责 |
|------|------|
| `mapping/layers.py` | 新增 Conv1d, ConvTranspose2d, BatchNorm1d, BatchNorm2d（与现有 Conv2d/Linear 同文件） |
| `mapping/resblock.py` | 新建：trunk 级 ResBlock 容器 |
| `mapping/__init__.py` | 更新公共 API 导出 |
| `tests/test_layers_ext.py` | 新建：Conv1d, ConvTranspose2d, BatchNorm1d/2d 测试 |
| `tests/test_resblock.py` | 新建：ResBlock 测试 |
| `tests/test_integration_phase3.py` | 新建：Phase 3 集成测试（Sequential + ResBlock） |

---

### Task 1: mapping.Conv1d

**Files:**
- Modify: `mapping/layers.py`
- Test: `tests/test_layers_ext.py`

**Interfaces:**
- Consumes: `MappingLayer` 基类 (`mapping/base.py`), `F.conv1d`
- Produces: `mapping.layers.Conv1d` — param_spec `{'weight': (C_out, C_in, k), 'bias': (C_out,)}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layers_ext.py`:

```python
import pytest
import torch
import torch.nn.functional as F
from mapping.base import Generator, MappingLayer
from mapping.layers import Conv1d


class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestConv1d:
    def test_param_spec_auto_deduced(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16, 4, 3)
        assert layer.param_spec['bias'] == (16,)

    def test_param_spec_no_bias(self, device):
        layer = Conv1d(4, 16, 3, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        assert y.shape == (2, 16, 98)

    def test_forward_with_params(self, device):
        layer = Conv1d(4, 16, 3).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w = torch.randn(16, 4, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv1d(x, w, b)
        assert torch.allclose(y, expected)

    def test_flat_params_auto_reshape(self, device):
        layer = Conv1d(4, 16, 3).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w_flat = torch.randn(192, device=device)  # 16*4*3
        b_flat = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 16, 98)

    def test_stride_padding(self, device):
        layer = Conv1d(4, 16, 3, stride=2, padding=1).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w = torch.randn(16, 4, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv1d(x, w, b, stride=2, padding=1)
        assert torch.allclose(y, expected)

    def test_gradient_flows(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestConv1d -v`
Expected: FAIL with `ImportError: cannot import name 'Conv1d' from 'mapping.layers'`

- [ ] **Step 3: Implement Conv1d**

Append to `mapping/layers.py` after the `Linear` class:

```python
class Conv1d(MappingLayer):
    """1D 卷积映射层。

    init 签名对齐 torch.nn.Conv1d。param_spec 自动推导。

    Args:
        in_channels  (int): 输入通道数 C_in
        out_channels (int): 输出通道数 C_out
        kernel_size  (int): 卷积核尺寸 k
        stride       (int): 步长 (默认 1)
        padding      (int): 填充 (默认 0)
        dilation     (int): 膨胀 (默认 1)
        groups       (int): 分组卷积数 (默认 1)
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用）
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C_out, C_in, k)
        bias:   (C_out,)  (仅 bias=True 时)
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
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        self.param_spec = {'weight': (out_channels, in_channels, kernel_size)}
        if bias:
            self.param_spec['bias'] = (out_channels,)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.conv1d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestConv1d -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add mapping/layers.py tests/test_layers_ext.py
git commit -m "feat: add mapping.Conv1d trunk layer"
```

---

### Task 2: mapping.ConvTranspose2d

**Files:**
- Modify: `mapping/layers.py`
- Test: `tests/test_layers_ext.py`

**Interfaces:**
- Consumes: `MappingLayer` 基类, `F.conv_transpose2d`
- Produces: `mapping.layers.ConvTranspose2d` — param_spec `{'weight': (C_in, C_out, kh, kw), 'bias': (C_out,)}`（注意 torch 转置卷积权重布局）

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layers_ext.py`:

```python
from mapping.layers import ConvTranspose2d


class TestConvTranspose2d:
    def test_param_spec_auto_deduced(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16, 8, 3, 3)
        assert layer.param_spec['bias'] == (8,)

    def test_param_spec_no_bias(self, device):
        layer = ConvTranspose2d(16, 8, 3, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 9, 9)

    def test_forward_with_params(self, device):
        layer = ConvTranspose2d(16, 8, 3).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w = torch.randn(16, 8, 3, 3, device=device)
        b = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv_transpose2d(x, w, b)
        assert torch.allclose(y, expected)

    def test_stride_padding_output_padding(self, device):
        layer = ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w = torch.randn(16, 8, 3, 3, device=device)
        b = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)
        assert torch.allclose(y, expected)
        assert y.shape == (2, 8, 14, 14)

    def test_flat_params_auto_reshape(self, device):
        layer = ConvTranspose2d(16, 8, 3).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w_flat = torch.randn(1152, device=device)  # 16*8*3*3
        b_flat = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 8, 9, 9)

    def test_gradient_flows(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestConvTranspose2d -v`
Expected: FAIL with `ImportError: cannot import name 'ConvTranspose2d' from 'mapping.layers'`

- [ ] **Step 3: Implement ConvTranspose2d**

Append to `mapping/layers.py` after `Conv1d`:

```python
class ConvTranspose2d(MappingLayer):
    """2D 转置卷积映射层。

    init 签名对齐 torch.nn.ConvTranspose2d。param_spec 自动推导。
    注意 torch 转置卷积权重布局为 (C_in, C_out, kh, kw)。

    Args:
        in_channels  (int): 输入通道数 C_in
        out_channels (int): 输出通道数 C_out
        kernel_size  (int | tuple): 卷积核尺寸 (kh, kw)
        stride       (int | tuple): 步长 (默认 1)
        padding      (int | tuple): 填充 (默认 0)
        output_padding (int | tuple): 输出填充 (默认 0)
        dilation     (int | tuple): 膨胀 (默认 1)
        groups       (int): 分组卷积数 (默认 1)
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用）
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C_in, C_out, kh, kw)
        bias:   (C_out,)  (仅 bias=True 时)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        kh, kw = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        self.param_spec = {'weight': (in_channels, out_channels, kh, kw)}
        if bias:
            self.param_spec['bias'] = (out_channels,)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding,
            self.groups, self.dilation,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestConvTranspose2d -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add mapping/layers.py tests/test_layers_ext.py
git commit -m "feat: add mapping.ConvTranspose2d trunk layer"
```

---

### Task 3: mapping.BatchNorm1d / BatchNorm2d

**Files:**
- Modify: `mapping/layers.py`
- Test: `tests/test_layers_ext.py`

**Interfaces:**
- Consumes: `MappingLayer` 基类, `F.batch_norm`
- Produces: `mapping.layers.BatchNorm1d`, `mapping.layers.BatchNorm2d` — param_spec `{'weight': (num_features,), 'bias': (num_features,)}`；running_mean/running_var/num_batches_tracked 为 buffer

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layers_ext.py`:

```python
from mapping.layers import BatchNorm1d, BatchNorm2d


class TestBatchNorm2d:
    def test_param_spec(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16,)
        assert layer.param_spec['bias'] == (16,)

    def test_buffers_registered(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'running_mean' in dict(layer.named_buffers())
        assert 'running_var' in dict(layer.named_buffers())
        assert 'num_batches_tracked' in dict(layer.named_buffers())
        assert layer.running_mean.shape == (16,)
        assert layer.running_var.shape == (16,)

    def test_forward_train_mode(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)

    def test_forward_eval_mode(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.eval()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)

    def test_forward_with_params(self, device):
        layer = BatchNorm2d(16).to(device)
        layer.eval()
        x = torch.randn(4, 16, 8, 8, device=device)
        w = torch.ones(16, device=device)
        b = torch.zeros(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.batch_norm(
            x, layer.running_mean, layer.running_var, w, b, False, 0.1, 1e-5
        )
        assert torch.allclose(y, expected)

    def test_running_stats_updated_in_train(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        _ = layer(x)
        assert not torch.allclose(
            layer.running_mean, torch.zeros(16, device=device)
        )

    def test_gradient_flows(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_custom_eps_momentum(self, device):
        layer = BatchNorm2d(16, eps=1e-3, momentum=0.2,
                            generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.eps == 1e-3
        assert layer.momentum == 0.2


class TestBatchNorm1d:
    def test_param_spec(self, device):
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (32,)
        assert layer.param_spec['bias'] == (32,)

    def test_forward_2d_input(self, device):
        """BatchNorm1d 接受 (N, C) 输入。"""
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(8, 32, device=device)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_forward_3d_input(self, device):
        """BatchNorm1d 接受 (N, C, L) 输入。"""
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 32, 100, device=device)
        y = layer(x)
        assert y.shape == (4, 32, 100)

    def test_forward_with_params(self, device):
        layer = BatchNorm1d(32).to(device)
        layer.eval()
        x = torch.randn(8, 32, device=device)
        w = torch.ones(32, device=device)
        b = torch.zeros(32, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.batch_norm(
            x, layer.running_mean, layer.running_var, w, b, False, 0.1, 1e-5
        )
        assert torch.allclose(y, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestBatchNorm2d tests/test_layers_ext.py::TestBatchNorm1d -v`
Expected: FAIL with `ImportError: cannot import name 'BatchNorm2d' from 'mapping.layers'`

- [ ] **Step 3: Implement BatchNorm2d and BatchNorm1d**

Append to `mapping/layers.py`:

```python
class BatchNorm2d(MappingLayer):
    """2D 批归一化映射层。

    weight/bias (gamma/beta) 由 generator 生成；
    running_mean/running_var/num_batches_tracked 作为 buffer 保留在层内。

    Args:
        num_features (int): 特征通道数 C
        eps          (float): 数值稳定常数 (默认 1e-5)
        momentum     (float): 运行统计更新动量 (默认 0.1)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用）
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C,)
        bias:   (C,)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        self.param_spec = {'weight': (num_features,), 'bias': (num_features,)}

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.batch_norm(
            x, self.running_mean, self.running_var, w, b,
            self.training, self.momentum, self.eps,
        )


class BatchNorm1d(MappingLayer):
    """1D 批归一化映射层。

    接受 (N, C) 或 (N, C, L) 输入。weight/bias 由 generator 生成；
    running_mean/running_var/num_batches_tracked 作为 buffer 保留在层内。

    Args:
        num_features (int): 特征数 C
        eps          (float): 数值稳定常数 (默认 1e-5)
        momentum     (float): 运行统计更新动量 (默认 0.1)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用）
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C,)
        bias:   (C,)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        self.param_spec = {'weight': (num_features,), 'bias': (num_features,)}

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.batch_norm(
            x, self.running_mean, self.running_var, w, b,
            self.training, self.momentum, self.eps,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_layers_ext.py::TestBatchNorm2d tests/test_layers_ext.py::TestBatchNorm1d -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add mapping/layers.py tests/test_layers_ext.py
git commit -m "feat: add mapping.BatchNorm1d and BatchNorm2d trunk layers"
```

---

### Task 4: mapping.ResBlock

**Files:**
- Create: `mapping/resblock.py`
- Test: `tests/test_resblock.py`

**Interfaces:**
- Consumes: `MappingLayer` (`mapping/base.py`), `Conv2d` (`mapping/layers.py`), `_prod` (`mapping/base.py`)
- Produces: `mapping.resblock.ResBlock` — LWT 模式内部各层自带 generator；SLVT 模式 param_spec 为聚合 flat `{'weight': (total_w,), 'bias': (total_b,)}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resblock.py`:

```python
import pytest
import torch
import torch.nn.functional as F
from mapping.base import Generator, MappingLayer, _prod
from mapping.layers import Conv2d
from mapping.resblock import ResBlock


class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestResBlockLWT:
    """LWT 模式：内部各层自带 generator。"""

    def test_is_mapping_layer(self, device):
        block = ResBlock(16, 16, generator_cls=SimpleGen, z_dim=32).to(device)
        assert isinstance(block, MappingLayer)

    def test_forward_same_channels(self, device):
        block = ResBlock(16, 16, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 16, 8, 8)

    def test_forward_channel_change_enables_shortcut(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert block.use_shortcut
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 32, 8, 8)

    def test_forward_stride_change_enables_shortcut(self, device):
        block = ResBlock(16, 16, stride=2, generator_cls=SimpleGen, z_dim=32).to(device)
        assert block.use_shortcut
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 16, 4, 4)

    def test_internal_layers_have_generators(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert hasattr(block.conv1, 'generator')
        assert hasattr(block.conv2, 'generator')
        assert hasattr(block.shortcut, 'generator')

    def test_gradient_flows_to_all_z(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        y.sum().backward()
        assert block.conv1.generator.z.grad is not None
        assert block.conv2.generator.z.grad is not None
        assert block.shortcut.generator.z.grad is not None


class TestResBlockSLVT:
    """SLVT 模式：纯形状层，聚合 param_spec。"""

    def test_no_generator_on_block(self, device):
        block = ResBlock(16, 16).to(device)
        assert not hasattr(block, 'generator')

    def test_aggregated_param_spec_same_channels(self, device):
        block = ResBlock(16, 16).to(device)
        conv1_w = 16 * 16 * 3 * 3
        conv2_w = 16 * 16 * 3 * 3
        total_w = conv1_w + conv2_w
        total_b = 16 + 16
        assert block.param_spec == {'weight': (total_w,), 'bias': (total_b,)}

    def test_aggregated_param_spec_with_shortcut(self, device):
        block = ResBlock(16, 32).to(device)
        conv1_w = 32 * 16 * 3 * 3
        conv2_w = 32 * 32 * 3 * 3
        shortcut_w = 32 * 16 * 1 * 1
        total_w = conv1_w + conv2_w + shortcut_w
        total_b = 32 + 32 + 32
        assert block.param_spec == {'weight': (total_w,), 'bias': (total_b,)}

    def test_forward_with_params(self, device):
        block = ResBlock(16, 16).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.randn(total_w, device=device)
        b = torch.randn(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert y.shape == (2, 16, 8, 8)

    def test_forward_with_params_shortcut(self, device):
        block = ResBlock(16, 32, stride=2).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.randn(total_w, device=device)
        b = torch.randn(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert y.shape == (2, 32, 4, 4)

    def test_no_bias(self, device):
        block = ResBlock(16, 16, bias=False).to(device)
        assert 'bias' not in block.param_spec
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        w = torch.randn(total_w, device=device)
        y = block.forward_with_params(x, w, None)
        assert y.shape == (2, 16, 8, 8)

    def test_residual_property(self, device):
        """零权重时输出等于输入（同通道无 stride）。"""
        block = ResBlock(16, 16).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.zeros(total_w, device=device)
        b = torch.zeros(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert torch.allclose(y, x, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_resblock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mapping.resblock'`

- [ ] **Step 3: Implement ResBlock**

Create `mapping/resblock.py`:

```python
"""Trunk 级残差块容器。"""

import torch
import torch.nn.functional as F

from mapping.base import MappingLayer, _prod
from mapping.layers import Conv2d


class ResBlock(MappingLayer):
    """主干级残差块：两个 Conv2d + 跳连。

    双模式：
    - LWT：传 generator_cls，内部各层各自带 generator，forward() 直接调用
    - SLVT：不传 generator_cls，纯形状层，param_spec 为聚合 flat，
      forward_with_params 收到整段切片后内部按边界二次切片

    通道数或空间尺寸变化时自动启用 1x1 shortcut 卷积。

    Args:
        in_channels  (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size  (int): 卷积核尺寸 (默认 3)
        stride       (int): 第一个卷积的步长 (默认 1)
        padding      (int | None): 填充，默认 kernel_size // 2
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        **generator_kwargs: 透传给 generator 构造函数的参数
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
        generator_cls=None,
        **generator_kwargs,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.use_shortcut = (in_channels != out_channels) or (stride != 1)
        self.has_bias = bias

        gen_kw = dict(generator_cls=generator_cls, **generator_kwargs) if generator_cls else {}

        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias, **gen_kw,
        )
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=bias, **gen_kw,
        )
        if self.use_shortcut:
            self.shortcut = Conv2d(
                in_channels, out_channels, 1,
                stride=stride, bias=bias, **gen_kw,
            )

        if generator_cls is None:
            self._build_aggregated_spec(bias)

    def _build_aggregated_spec(self, bias: bool) -> None:
        """SLVT 模式：聚合内部所有参数层的 param_spec 为 flat。"""
        layers = [self.conv1, self.conv2]
        if self.use_shortcut:
            layers.append(self.shortcut)

        w_total = sum(_prod(l.param_spec['weight']) for l in layers)
        self.param_spec = {'weight': (w_total,)}

        if bias:
            b_total = sum(_prod(l.param_spec['bias']) for l in layers)
            self.param_spec['bias'] = (b_total,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LWT 入口：内部各层用自己的 generator。"""
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.use_shortcut:
            identity = self.shortcut(x)
        return out + identity

    def forward_with_params(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        """SLVT 入口：接收聚合 flat 参数，内部二次切片。"""
        offset_w = 0
        offset_b = 0

        def _slice(layer):
            nonlocal offset_w, offset_b
            ws = _prod(layer.param_spec['weight'])
            w_slice = w[offset_w:offset_w + ws]
            offset_w += ws
            b_slice = None
            if self.has_bias and b is not None:
                bs = _prod(layer.param_spec['bias'])
                b_slice = b[offset_b:offset_b + bs]
                offset_b += bs
            return w_slice, b_slice

        identity = x
        w1, b1 = _slice(self.conv1)
        out = F.relu(self.conv1.forward_with_params(x, w1, b1))

        w2, b2 = _slice(self.conv2)
        out = self.conv2.forward_with_params(out, w2, b2)

        if self.use_shortcut:
            ws, bs = _slice(self.shortcut)
            identity = self.shortcut.forward_with_params(x, ws, bs)

        return out + identity
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_resblock.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add mapping/resblock.py tests/test_resblock.py
git commit -m "feat: add mapping.ResBlock trunk-level residual container"
```

---

### Task 5: Public API exports + integration tests

**Files:**
- Modify: `mapping/__init__.py`
- Create: `tests/test_integration_phase3.py`

**Interfaces:**
- Consumes: 所有 Task 1-4 产出的层
- Produces: `mapping` 包顶层导出 Conv1d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, ResBlock

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_integration_phase3.py`:

```python
import torch
import torch.nn.functional as F
from mapping import (
    Conv1d, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d,
    Linear, ResBlock, Sequential, Generator,
)
from mapping.generator import Linear as GenLinear


class MyGen(Generator):
    def __init__(self, param_spec, z_dim=64, hidden_dim=128):
        super().__init__(param_spec, z_dim=z_dim)
        self.body = torch.nn.Sequential(
            GenLinear(z_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.w_head = torch.nn.Linear(hidden_dim, self.w_size)
        self.b_head = (
            torch.nn.Linear(hidden_dim, self.b_size) if self.b_size > 0 else None
        )

    def forward(self):
        h = self.body(self.z)
        w = self.w_head(h).reshape(self.w_shape)
        b = self.b_head(h).reshape(self.b_shape) if self.b_head is not None else None
        return w, b


class TestSequentialWithResBlock:
    """ResBlock 作为纯形状层放入 Sequential。"""

    def test_slvt_with_resblock(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            ResBlock(16, 16),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(16 * 14 * 14, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_slvt_with_resblock_channel_change(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            ResBlock(16, 32, stride=2),
            torch.nn.Flatten(1),
            Linear(32 * 14 * 14, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_gradient_flows_through_resblock(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            ResBlock(16, 16),
            torch.nn.Flatten(1),
            Linear(16 * 28 * 28, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        y.sum().backward()
        assert net.generator.z.grad is not None


class TestLWTWithNewLayers:
    """LWT 模式使用新层类型。"""

    def test_lwt_conv1d(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        assert y.shape == (2, 16, 98)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_conv_transpose2d(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 9, 9)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_batchnorm2d(self, device):
        layer = BatchNorm2d(16, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_resblock(self, device):
        block = ResBlock(16, 32, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 32, 8, 8)
        y.sum().backward()
        assert block.conv1.generator.z.grad is not None


class TestSequentialWithBatchNorm:
    """BatchNorm 在 Sequential 中的行为。"""

    def test_slvt_with_batchnorm(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            Linear(16 * 28 * 28, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        net.train()
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_integration_phase3.py -v`
Expected: FAIL with `ImportError: cannot import name 'Conv1d' from 'mapping'`

- [ ] **Step 3: Update mapping/__init__.py exports**

Replace `mapping/__init__.py` with:

```python
"""Mapping 推理框架 - 参数生成 + 主干网络的前向推理框架。"""

from mapping.base import Generator, MappingLayer
from mapping.generator.lrd import LRDLayer
from mapping.layers import BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, ConvTranspose2d, Linear
from mapping.resblock import ResBlock
from mapping.sequential import Sequential

__all__ = [
    'Generator',
    'MappingLayer',
    'LRDLayer',
    'Conv1d',
    'Conv2d',
    'ConvTranspose2d',
    'BatchNorm1d',
    'BatchNorm2d',
    'Linear',
    'ResBlock',
    'Sequential',
]
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_integration_phase3.py -v`
Expected: 8 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest -v`
Expected: All tests pass (previous 146 + new tests)

- [ ] **Step 6: Run ruff check**

Run: `uv run ruff check mapping/ tests/test_layers_ext.py tests/test_resblock.py tests/test_integration_phase3.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add mapping/__init__.py tests/test_integration_phase3.py
git commit -m "feat: export Phase 3 layers and add integration tests"
```
