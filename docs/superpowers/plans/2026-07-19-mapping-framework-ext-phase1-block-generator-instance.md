# Mapping 框架扩展阶段 1：generator.Block 基类 + generator_instance 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `mapping.generator.Block` 元类基类（继承后写法与 torch 一致，自动 init_weights + 递归冻结参数），将现有 Linear/Conv1d/Conv2d 重构为 Block 子类，并为 MappingLayer 增加 `generator_instance` 扩展（GitHub issue #14）。

**Architecture:** `_BlockMeta` 元类在实例 `__init__` 结束后自动调用 `init_weights()`（默认 no-op，叶子块重载为 kaiming uniform）→ `_freeze()`（递归 `requires_grad_(False)`）。叶子子块不再手动 freeze / 手动调 init_weights。`MappingLayer._set_generator` 统一处理 generator_cls / generator_instance 互斥与 param_spec 校验。

**Tech Stack:** Python 3 + PyTorch（nn.Module 元类机制）+ pytest

## Global Constraints

- 测试命令：`/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`（**禁止 `uv run`**）
- 测试必须使用 `tests/conftest.py` 的 `device` fixture；**禁止** `.cpu()`、`map_location='cpu'`、`device='cuda'` 硬编码
- **禁止改动 `mapping_network/` 包内任何文件**
- 代码带类型注解（对齐现有 `mapping/` 风格）
- 对外行为不变：`generator.Linear/Conv1d/Conv2d` 的 init 签名、初始化方法、冻结语义与现有测试 `tests/test_generator_blocks.py` 全部保持通过
- 相关设计文档：`docs/superpowers/specs/2026-07-19-mapping-framework-extension-design.md` §2

---

### Task 1: `mapping.generator.Block` 元类基类

**Files:**
- Create: `mapping/generator/block.py`
- Test: `tests/test_generator_block.py`

**Interfaces:**
- Consumes: 无（仅 torch.nn）
- Produces:
  - `class Block(nn.Module, metaclass=_BlockMeta)`：组合积木基类
    - `init_weights(self) -> None`：初始化钩子，默认 no-op；子类可重载
    - `_freeze(self) -> None`：递归冻结全部参数（`requires_grad_(False)`）
  - 元类行为：实例 `__init__` 结束后自动依次调用 `init_weights()` → `_freeze()`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_generator_block.py`：

```python
import torch
import torch.nn as nn
from mapping.generator.block import Block


class TestBlock:
    def test_auto_freeze_parameters(self, device):
        """Block 构造完成后所有参数自动 requires_grad=False（包括普通 torch 子模块）。"""

        class MyBlock(Block):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

        b = MyBlock().to(device)
        params = list(b.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_init_weights_called_once_after_init(self, device):
        """init_weights 在 __init__ 结束后被自动调用一次。"""

        calls = []

        class MyBlock(Block):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3))

            def init_weights(self) -> None:
                calls.append(1)
                nn.init.zeros_(self.weight)

        b = MyBlock().to(device)
        assert calls == [1]
        assert torch.all(b.weight == 0)

    def test_default_init_weights_noop(self, device):
        """默认 init_weights 为 no-op，不改动参数值。"""

        class MyBlock(Block):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3))

        b = MyBlock().to(device)
        assert torch.all(b.weight == 1)

    def test_nested_block_freeze_covers_descendants(self, device):
        """组合块嵌套时，外层冻结覆盖所有后代参数（幂等）。"""

        class Inner(Block):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3))

        class Outer(Block):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.weight = nn.Parameter(torch.ones(3))

        o = Outer().to(device)
        for p in o.parameters():
            assert not p.requires_grad

    def test_forward_composition_like_torch(self, device):
        """组合块 forward 像 torch 一样直接调用子模块（含跳连）。"""

        class ResBlock(Block):
            def __init__(self, dim: int):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.fc2(self.relu(self.fc1(x)))

        block = ResBlock(8).to(device)
        x = torch.randn(2, 8, device=device)
        y = block(x)
        assert y.shape == (2, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_block.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mapping.generator.block'`

- [ ] **Step 3: Write minimal implementation**

创建 `mapping/generator/block.py`：

```python
"""generator 积木基类：Block。"""

import torch.nn as nn


class _BlockMeta(type):
    """Block 元类：__init__ 结束后自动执行 init_weights() 并冻结全部参数。"""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.init_weights()
        instance._freeze()
        return instance


class Block(nn.Module, metaclass=_BlockMeta):
    """可组合的 generator 积木基类。

    继承本类后写法与 torch.nn.Module 完全一致：在 __init__ 中创建
    子模块，在 forward 中组合调用（支持残差等任意结构）。构造结束
    后框架自动执行 init_weights() 并递归冻结全部参数
    (requires_grad_(False))，用户无需手动处理。

    子类可重载 init_weights() 自定义初始化；默认为 no-op
    （组合块不重新初始化已就位的子块）。
    """

    def init_weights(self) -> None:
        """初始化钩子，默认 no-op。子类可重载。"""

    def _freeze(self) -> None:
        """递归冻结全部参数（幂等）。"""
        for p in self.parameters():
            p.requires_grad_(False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_block.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add mapping/generator/block.py tests/test_generator_block.py
git commit -m "feat: add mapping.generator.Block composable base with metaclass auto-freeze"
```

---

### Task 2: 现有子块重构为 Block 子类 + 导出

**Files:**
- Modify: `mapping/generator/linear.py`
- Modify: `mapping/generator/conv.py`
- Modify: `mapping/generator/__init__.py`
- Test: `tests/test_generator_blocks.py`（既有测试，须全部保持通过）
- Test: `tests/test_generator_block.py`（追加集成断言）

**Interfaces:**
- Consumes: Task 1 的 `Block`（`mapping/generator/block.py`）
- Produces:
  - `generator.Linear` / `generator.Conv1d` / `generator.Conv2d`：签名与行为不变，成为 `Block` 子类；参数创建后由元类统一 init + freeze
  - `mapping.generator.__init__` 新增导出 `Block`（`__all__ = ['Block', 'Linear', 'Conv1d', 'Conv2d', 'LRDLayer']`）

- [ ] **Step 1: Write the failing test**

在 `tests/test_generator_block.py` 末尾追加：

```python
class TestLeafBlocksAreBlocks:
    def test_linear_is_block(self, device):
        """generator.Linear 是 Block 子类，且行为不变。"""
        from mapping.generator import Block, Linear

        layer = Linear(10, 20).to(device)
        assert isinstance(layer, Block)
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad
        # 已初始化（非全零）
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_conv_blocks_are_blocks(self, device):
        """generator.Conv1d / Conv2d 是 Block 子类。"""
        from mapping.generator import Block, Conv1d, Conv2d

        c1 = Conv1d(3, 16, 3).to(device)
        c2 = Conv2d(3, 16, 3).to(device)
        assert isinstance(c1, Block)
        assert isinstance(c2, Block)
        assert not c1.weight.requires_grad
        assert not c2.weight.requires_grad

    def test_custom_init_weights_still_works(self, device):
        """用户重载 init_weights 在重构后仍生效（由元类调用）。"""
        from mapping.generator import Linear

        class CustomLinear(Linear):
            def init_weights(self) -> None:
                torch.nn.init.ones_(self.weight)
                if self.bias is not None:
                    torch.nn.init.zeros_(self.bias)

        layer = CustomLinear(10, 20).to(device)
        assert torch.allclose(layer.weight, torch.ones_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_residual_block_with_generator_subblocks(self, device):
        """用 generator 子块组合残差块：无需手动 freeze/init。"""
        import torch.nn as nn
        from mapping.generator import Block, Conv2d

        class ConvResBlock(Block):
            def __init__(self, channels: int):
                super().__init__()
                self.conv1 = Conv2d(channels, channels, 3, padding=1)
                self.conv2 = Conv2d(channels, channels, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.conv2(self.relu(self.conv1(x)))

        block = ConvResBlock(8).to(device)
        for p in block.parameters():
            assert not p.requires_grad
        x = torch.randn(2, 8, 10, 10, device=device)
        y = block(x)
        assert y.shape == (2, 8, 10, 10)
```

（文件顶部已 import torch / nn / Block，注意追加处需要的 `import torch.nn as nn` 已在文件顶部存在则复用。）

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_block.py -v`
Expected: 新增的 4 个测试 FAIL（`ImportError: cannot import name 'Block' from 'mapping.generator'` 或 isinstance 断言失败）

- [ ] **Step 3: Write minimal implementation**

3a. 重写 `mapping/generator/linear.py`：

```python
"""固定随机参数 Linear 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.generator.block import Block


class Linear(Block):
    """固定随机参数的线性层子块。

    init 签名对齐 torch.nn.Linear。内部参数在构造时随机初始化
    并设为 requires_grad=False（由 Block 元类自动完成）。
    默认采用论文方法初始化，用户可重载 init_weights() 自定义。

    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        # init_weights() 与参数冻结由 Block 元类在构造结束后自动完成

    def init_weights(self) -> None:
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

3b. 重写 `mapping/generator/conv.py`：

```python
"""固定随机参数 Conv1d / Conv2d 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.generator.block import Block


class _ConvNd(Block):
    """Conv 子块基类，共享 init_weights 逻辑。"""

    def init_weights(self) -> None:
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
    并设为 requires_grad=False（由 Block 元类自动完成）。

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
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # init_weights() 与参数冻结由 Block 元类在构造结束后自动完成

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class Conv2d(_ConvNd):
    """固定随机参数的二维卷积子块。

    init 签名对齐 torch.nn.Conv2d。内部参数在构造时随机初始化
    并设为 requires_grad=False（由 Block 元类自动完成）。

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
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # init_weights() 与参数冻结由 Block 元类在构造结束后自动完成

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )
```

3c. 更新 `mapping/generator/__init__.py`：

```python
from mapping.generator.block import Block
from mapping.generator.linear import Linear
from mapping.generator.conv import Conv1d, Conv2d
from mapping.generator.lrd import LRDLayer

__all__ = ['Block', 'Linear', 'Conv1d', 'Conv2d', 'LRDLayer']
```

- [ ] **Step 4: Run tests to verify they pass（新增 + 既有全量）**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_block.py tests/test_generator_blocks.py -v`
Expected: 全部通过（新增 9 个 + 既有全部）

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`
Expected: 全量通过（含 test_integration.py 中对 GenLinear 的使用）

- [ ] **Step 5: Commit**

```bash
git add mapping/generator/linear.py mapping/generator/conv.py mapping/generator/__init__.py tests/test_generator_block.py
git commit -m "refactor: make generator Linear/Conv1d/Conv2d subclasses of Block"
```

---

### Task 3: MappingLayer `generator_instance` 扩展

**Files:**
- Modify: `mapping/base.py`（MappingLayer 新增 `_set_generator`）
- Modify: `mapping/layers.py`（Conv2d/Linear 新增 `generator_instance` 参数）
- Test: `tests/test_layers.py`（追加 `TestGeneratorInstance`）

**Interfaces:**
- Consumes: 现有 `Generator`（`mapping/base.py`，暴露 `w_shape`/`b_shape`）
- Produces:
  - `MappingLayer._set_generator(self, generator_cls: type[Generator] | None, generator_instance: Generator | None, generator_kwargs: dict) -> None`
    - 两者同传 → `ValueError`（互斥）
    - `generator_instance` 非 `Generator` 实例 → `TypeError`
    - `generator_instance.w_shape != param_spec['weight']` 或 `b_shape != param_spec.get('bias')` → `ValueError`
    - 合法时设置 `self.generator`
  - `Conv2d(..., generator_cls=None, generator_instance=None, **generator_kwargs)`
  - `Linear(..., generator_cls=None, generator_instance=None, **generator_kwargs)`
  - 语义 = 权重捆绑：挂同一实例的层获得完全相同的 `(weight, bias)`；`mapping.Sequential` 现有检查（层自带 generator 则报错）自动覆盖该情形

- [ ] **Step 1: Write the failing test**

在 `tests/test_layers.py` 末尾追加（文件顶部需补充 `import pytest`）：

```python
class TestGeneratorInstance:
    def test_weight_tying_shared_instance(self, device):
        """两层挂同一 generator 实例：输出参数完全相同（权重捆绑）。"""
        spec = {'weight': (4, 3), 'bias': (4,)}
        gen = SimpleGen(spec, z_dim=8).to(device)
        l1 = Linear(3, 4, generator_instance=gen).to(device)
        l2 = Linear(3, 4, generator_instance=gen).to(device)

        assert l1.generator is gen
        assert l2.generator is gen

        x = torch.randn(2, 3, device=device)
        w1, b1 = l1.generator()
        w2, b2 = l2.generator()
        assert torch.equal(w1, w2)
        assert torch.equal(b1, b2)
        assert l1(x).shape == (2, 4)

    def test_param_spec_mismatch_raises(self, device):
        """generator_instance 的 param_spec 与层不一致时报 ValueError。"""
        gen = SimpleGen({'weight': (5, 3), 'bias': (5,)}, z_dim=8).to(device)
        with pytest.raises(ValueError):
            Linear(3, 4, generator_instance=gen)

    def test_bias_mismatch_raises(self, device):
        """无 bias 的 instance 传给有 bias 的层时报 ValueError。"""
        gen = SimpleGen({'weight': (4, 3)}, z_dim=8).to(device)
        with pytest.raises(ValueError):
            Linear(3, 4, bias=True, generator_instance=gen)

    def test_mutual_exclusion_raises(self, device):
        """generator_cls 与 generator_instance 同传时报 ValueError。"""
        gen = SimpleGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        with pytest.raises(ValueError):
            Linear(3, 4, generator_cls=SimpleGen, generator_instance=gen, z_dim=8)

    def test_non_generator_instance_raises(self, device):
        """传入非 Generator 实例时报 TypeError。"""
        with pytest.raises(TypeError):
            Linear(3, 4, generator_instance=torch.nn.Linear(3, 4))

    def test_conv2d_generator_instance(self, device):
        """Conv2d 同样支持 generator_instance。"""
        spec = {'weight': (16, 3, 3, 3), 'bias': (16,)}
        gen = SimpleGen(spec, z_dim=8).to(device)
        layer = Conv2d(3, 16, 3, generator_instance=gen).to(device)
        x = torch.randn(2, 3, 10, 10, device=device)
        assert layer(x).shape == (2, 16, 8, 8)
```

注意：`SimpleGen` 复用 `tests/test_layers.py` 文件顶部已定义的类（`__init__(self, param_spec, z_dim=32, **kwargs)`）。

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_layers.py::TestGeneratorInstance -v`
Expected: FAIL（`TypeError: __init__() got an unexpected keyword argument 'generator_instance'`）

- [ ] **Step 3: Write minimal implementation**

3a. `mapping/base.py`：在 `MappingLayer` 类中（`_resolve` 方法之前）新增：

```python
    def _set_generator(
        self,
        generator_cls: type[Generator] | None,
        generator_instance: Generator | None,
        generator_kwargs: dict,
    ) -> None:
        """根据互斥规则设置 self.generator。

        - generator_cls 与 generator_instance 互斥，同传抛 ValueError
        - generator_instance 必须是 Generator 实例，否则 TypeError
        - generator_instance 的 w_shape/b_shape 必须与 self.param_spec 一致，
          否则 ValueError
        - 都不传则为纯形状层（SLVT，由 Sequential 供参）
        """
        if generator_cls is not None and generator_instance is not None:
            raise ValueError(
                'generator_cls 与 generator_instance 互斥，只能传其中一个'
            )
        if generator_instance is not None:
            if not isinstance(generator_instance, Generator):
                raise TypeError(
                    f'generator_instance 必须是 Generator 实例，'
                    f'得到 {type(generator_instance).__name__}'
                )
            expected_w = self.param_spec['weight']
            expected_b = self.param_spec.get('bias')
            if (
                generator_instance.w_shape != expected_w
                or generator_instance.b_shape != expected_b
            ):
                raise ValueError(
                    f'generator_instance 的 param_spec 与层推导的不一致: '
                    f'期望 weight={expected_w}, bias={expected_b}，'
                    f'实际 weight={generator_instance.w_shape}, '
                    f'bias={generator_instance.b_shape}'
                )
            self.generator = generator_instance
        elif generator_cls is not None:
            self.generator = generator_cls(self.param_spec, **generator_kwargs)
```

3b. `mapping/layers.py` `Conv2d.__init__` 签名与尾部修改：

```python
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
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
```

尾部将：

```python
        if generator_cls is not None:
            self.generator = generator_cls(self.param_spec, **generator_kwargs)
```

替换为：

```python
        self._set_generator(generator_cls, generator_instance, generator_kwargs)
```

docstring Args 中 `generator_cls` 行之后补充：

```
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用），
            与 generator_cls 互斥；param_spec 必须与层推导一致
```

3c. `mapping/layers.py` `Linear.__init__` 同样修改：签名在 `generator_cls` 后加 `generator_instance: Generator | None = None`，尾部同样替换为 `self._set_generator(generator_cls, generator_instance, generator_kwargs)`，docstring 同样补充。

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_layers.py -v`
Expected: 全部通过（新增 6 个 + 既有全部）

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`
Expected: 全量通过

- [ ] **Step 5: Commit**

```bash
git add mapping/base.py mapping/layers.py tests/test_layers.py
git commit -m "feat: add generator_instance support to MappingLayer (weight tying)"
```

---

## Self-Review 记录

- **Spec 覆盖**：§2.1 Block 基类 → Task 1；§2.2 现有子块重构 → Task 2；§2.4 generator_instance → Task 3。§2.3 预置积木块属 issue #15（阶段 2），不在本计划。
- **类型一致性**：`init_weights(self) -> None`、`_freeze(self) -> None`、`_set_generator(self, generator_cls, generator_instance, generator_kwargs) -> None` 在 Task 间一致；`SimpleGen(param_spec, z_dim=...)` 与 tests/test_layers.py 现有定义一致。
- **LRDLayer**：无参模块，按设计不改动。
