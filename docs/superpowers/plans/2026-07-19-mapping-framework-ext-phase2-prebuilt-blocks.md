# Mapping 框架扩展阶段 2：预置 generator 积木块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 `mapping.generator.Block` 实现预置积木块 `generator.MLP` / `generator.LinearResBlock` / `generator.ConvResBlock` / `generator.TransformerBlock`（GitHub issue #15）。

**Architecture:** 三个新模块文件（mlp.py / resblock.py / transformer.py），全部继承 `Block`（阶段 1 已实现：元类自动 init_weights + 递归冻结），内部组合 `generator.Linear` / `generator.Conv2d` 叶子块与普通 torch 模块。TransformerBlock 的 FFN 复用 MLP。

**Tech Stack:** Python 3 + PyTorch + pytest

## Global Constraints

- 测试命令：`/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`（**禁止 `uv run`**）
- 测试必须使用 `tests/conftest.py` 的 `device` fixture；**禁止** `.cpu()`、`map_location='cpu'`、`device='cuda'` 硬编码
- **禁止改动 `mapping_network/` 包内任何文件**
- 代码带类型注解（对齐现有 `mapping/` 风格）
- 所有新积木必须是 `Block` 子类，全部参数自动 `requires_grad=False`，用户可重载 `init_weights()` 自定义初始化
- 既有测试（121 个）全部保持通过
- 相关设计文档：`docs/superpowers/specs/2026-07-19-mapping-framework-extension-design.md` §2.3
- 前置（已实现）：`mapping/generator/block.py` 的 `Block`（`init_weights()` 默认 no-op、`_freeze()` 递归冻结）；`mapping/generator/linear.py` 的 `Linear`；`mapping/generator/conv.py` 的 `Conv2d`

---

### Task 1: `generator.MLP`

**Files:**
- Create: `mapping/generator/mlp.py`
- Modify: `mapping/generator/__init__.py`
- Test: `tests/test_generator_mlp.py`

**Interfaces:**
- Consumes: `Block`（mapping/generator/block.py）、`Linear`（mapping/generator/linear.py）
- Produces:
  - `MLP(sizes: list[int] | tuple[int, ...], act: type[nn.Module] = nn.ReLU)`
    - 结构：`Linear(sizes[i], sizes[i+1])` 串联，激活夹在相邻 Linear 之间（最后无激活）
    - `len(sizes) < 2` → `ValueError`
    - 属性：`self.sizes: list[int]`、`self.layers: nn.ModuleList`
    - `forward(x: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_generator_mlp.py`：

```python
import pytest
import torch
import torch.nn as nn
from mapping.generator import Block, Linear, MLP


class TestMLP:
    def test_is_block_and_frozen(self, device):
        """MLP 是 Block 子类，全部参数自动冻结。"""
        mlp = MLP([8, 16, 32]).to(device)
        assert isinstance(mlp, Block)
        params = list(mlp.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_structure(self, device):
        """len(sizes)-1 个 Linear，激活夹在中间，最后一个模块是 Linear。"""
        mlp = MLP([8, 16, 32, 4]).to(device)
        linears = [m for m in mlp.layers if isinstance(m, Linear)]
        acts = [m for m in mlp.layers if isinstance(m, nn.ReLU)]
        assert len(linears) == 3
        assert len(acts) == 2
        assert isinstance(mlp.layers[-1], Linear)
        assert mlp.layers[0].in_features == 8
        assert mlp.layers[-1].out_features == 4

    def test_forward_shape(self, device):
        mlp = MLP([8, 16, 32]).to(device)
        x = torch.randn(2, 8, device=device)
        assert mlp(x).shape == (2, 32)

    def test_forward_matches_manual(self, device):
        """输出等于逐层手动计算。"""
        mlp = MLP([8, 16, 32]).to(device)
        x = torch.randn(2, 8, device=device)
        h = x
        for layer in mlp.layers:
            h = layer(h)
        assert torch.equal(mlp(x), h)

    def test_sizes_too_short_raises(self, device):
        with pytest.raises(ValueError):
            MLP([8])

    def test_custom_activation(self, device):
        mlp = MLP([8, 16, 8], act=nn.GELU).to(device)
        acts = [m for m in mlp.layers if isinstance(m, nn.GELU)]
        assert len(acts) == 1

    def test_gradient_flows_to_input(self, device):
        """子块参数固定，但输入可梯度。"""
        mlp = MLP([8, 16, 8]).to(device)
        x = torch.randn(2, 8, device=device, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_mlp.py -v`
Expected: FAIL，`ImportError: cannot import name 'MLP' from 'mapping.generator'`

- [ ] **Step 3: Write minimal implementation**

3a. 创建 `mapping/generator/mlp.py`：

```python
"""预置积木块：MLP。"""

import torch
import torch.nn as nn

from mapping.generator.block import Block
from mapping.generator.linear import Linear


class MLP(Block):
    """多层感知机积木块。

    结构：Linear -> act -> ... -> Linear（最后无激活）。
    参数固定（Block 元类自动 init + freeze）。

    Args:
        sizes: 各层尺寸，如 [z_dim, 128, 256, out_dim]，至少 2 个
        act: 激活模块类（默认 nn.ReLU）
    """

    def __init__(
        self, sizes: list[int] | tuple[int, ...], act: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        if len(sizes) < 2:
            raise ValueError(f'MLP 至少需要 2 个层尺寸，得到 {sizes}')
        self.sizes = list(sizes)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

3b. `mapping/generator/__init__.py` 更新为：

```python
from mapping.generator.block import Block
from mapping.generator.linear import Linear
from mapping.generator.conv import Conv1d, Conv2d
from mapping.generator.lrd import LRDLayer
from mapping.generator.mlp import MLP

__all__ = ['Block', 'Linear', 'Conv1d', 'Conv2d', 'LRDLayer', 'MLP']
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_mlp.py -v`
Expected: 7 passed

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`
Expected: 全量通过（121 + 7 = 128）

- [ ] **Step 5: Commit**

```bash
git add mapping/generator/mlp.py mapping/generator/__init__.py tests/test_generator_mlp.py
git commit -m "feat: add mapping.generator.MLP prebuilt block"
```

---

### Task 2: `generator.LinearResBlock` / `generator.ConvResBlock`

**Files:**
- Create: `mapping/generator/resblock.py`
- Modify: `mapping/generator/__init__.py`
- Test: `tests/test_generator_resblock.py`

**Interfaces:**
- Consumes: `Block`、`Linear`（mapping/generator/linear.py）、`Conv2d`（mapping/generator/conv.py）
- Produces:
  - `LinearResBlock(dim: int)`：`x + fc2(act(fc1(x)))`，维度不变；属性 `fc1`/`fc2`/`act`
  - `ConvResBlock(channels: int, kernel_size: int = 3)`：`x + conv2(act(conv1(x)))`，`padding = kernel_size // 2`，通道与空间尺寸不变；属性 `conv1`/`conv2`/`act`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_generator_resblock.py`：

```python
import torch
import torch.nn as nn
from mapping.generator import Block, ConvResBlock, LinearResBlock


class TestLinearResBlock:
    def test_is_block_and_frozen(self, device):
        block = LinearResBlock(16).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        assert block(x).shape == (2, 16)

    def test_forward_matches_manual(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        expected = x + block.fc2(block.act(block.fc1(x)))
        assert torch.equal(block(x), expected)

    def test_residual_property(self, device):
        """fc2 输出为 0 时，输出严格等于输入（跳连恒等）。
        同时验证 init_weights 钩子在子块构造之后执行。"""

        class ZeroResBlock(LinearResBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.fc2.weight)
                nn.init.zeros_(self.fc2.bias)

        block = ZeroResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        assert torch.equal(block(x), x)

    def test_gradient_flows_to_input(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


class TestConvResBlock:
    def test_is_block_and_frozen(self, device):
        block = ConvResBlock(8).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        """通道与空间尺寸不变。"""
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert block(x).shape == (2, 8, 10, 10)

    def test_custom_kernel_size(self, device):
        block = ConvResBlock(8, kernel_size=5).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert block(x).shape == (2, 8, 10, 10)

    def test_forward_matches_manual(self, device):
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        expected = x + block.conv2(block.act(block.conv1(x)))
        assert torch.equal(block(x), expected)

    def test_residual_property(self, device):
        """conv2 输出为 0 时，输出严格等于输入。"""

        class ZeroConvResBlock(ConvResBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.conv2.weight)
                nn.init.zeros_(self.conv2.bias)

        block = ZeroConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert torch.equal(block(x), x)

    def test_gradient_flows_to_input(self, device):
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_resblock.py -v`
Expected: FAIL，`ImportError: cannot import name 'ConvResBlock' from 'mapping.generator'`

- [ ] **Step 3: Write minimal implementation**

3a. 创建 `mapping/generator/resblock.py`：

```python
"""预置积木块：残差块（linear 版 / conv 版）。"""

import torch
import torch.nn as nn

from mapping.generator.block import Block
from mapping.generator.conv import Conv2d
from mapping.generator.linear import Linear


class LinearResBlock(Block):
    """Linear 残差块：x + fc2(act(fc1(x)))。

    维度不变（跳连为恒等）。参数固定。

    Args:
        dim: 特征维度
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ConvResBlock(Block):
    """Conv2d 残差块：x + conv2(act(conv1(x)))。

    通道数与空间尺寸不变（padding = kernel_size // 2）。参数固定。

    Args:
        channels: 通道数
        kernel_size: 卷积核尺寸（默认 3）
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = Conv2d(channels, channels, kernel_size, padding=padding)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))
```

3b. `mapping/generator/__init__.py` 更新为：

```python
from mapping.generator.block import Block
from mapping.generator.linear import Linear
from mapping.generator.conv import Conv1d, Conv2d
from mapping.generator.lrd import LRDLayer
from mapping.generator.mlp import MLP
from mapping.generator.resblock import ConvResBlock, LinearResBlock

__all__ = [
    'Block', 'Linear', 'Conv1d', 'Conv2d', 'LRDLayer',
    'MLP', 'LinearResBlock', 'ConvResBlock',
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_resblock.py -v`
Expected: 11 passed

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`
Expected: 全量通过（128 + 11 = 139）

- [ ] **Step 5: Commit**

```bash
git add mapping/generator/resblock.py mapping/generator/__init__.py tests/test_generator_resblock.py
git commit -m "feat: add generator LinearResBlock and ConvResBlock prebuilt blocks"
```

---

### Task 3: `generator.TransformerBlock`

**Files:**
- Create: `mapping/generator/transformer.py`
- Modify: `mapping/generator/__init__.py`
- Test: `tests/test_generator_transformer.py`

**Interfaces:**
- Consumes: `Block`、`MLP`（Task 1，mapping/generator/mlp.py）
- Produces:
  - `TransformerBlock(dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0)`
    - pre-norm 结构：`x + attn(norm1(x))`，再 `x + ffn(norm2(x))`
    - 属性：`norm1`（nn.LayerNorm）、`attn`（nn.MultiheadAttention，batch_first=True）、`norm2`、`ffn`（`MLP([dim, int(dim*mlp_ratio), dim])`）
    - 输入形状 `(B, L, D)`，输出形状不变

- [ ] **Step 1: Write the failing test**

创建 `tests/test_generator_transformer.py`：

```python
import torch
import torch.nn as nn
from mapping.generator import Block, MLP, TransformerBlock


class TestTransformerBlock:
    def test_is_block_and_frozen(self, device):
        """TransformerBlock 是 Block 子类，全部参数（含 attn/LayerNorm）冻结。"""
        block = TransformerBlock(16, 4).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        block = TransformerBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device)  # (B, L, D)
        assert block(x).shape == (2, 5, 16)

    def test_deterministic_without_dropout(self, device):
        """dropout=0 时两次前向结果一致。"""
        block = TransformerBlock(16, 4, dropout=0.0).to(device)
        x = torch.randn(2, 5, 16, device=device)
        assert torch.equal(block(x), block(x))

    def test_residual_property(self, device):
        """attn 输出投影与 ffn 末层置零时，输出等于输入（跳连恒等）。
        同时验证 init_weights 钩子在子模块构造之后执行。"""

        class ZeroBlock(TransformerBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.attn.out_proj.weight)
                nn.init.zeros_(self.attn.out_proj.bias)
                last = self.ffn.layers[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

        block = ZeroBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device)
        assert torch.allclose(block(x), x)

    def test_mlp_ratio(self, device):
        """FFN 隐藏层维度 = int(dim * mlp_ratio)。"""
        block = TransformerBlock(16, 4, mlp_ratio=2.0).to(device)
        assert isinstance(block.ffn, MLP)
        assert block.ffn.layers[0].out_features == 32
        assert block.ffn.layers[-1].out_features == 16

    def test_gradient_flows_to_input(self, device):
        block = TransformerBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_transformer.py -v`
Expected: FAIL，`ImportError: cannot import name 'TransformerBlock' from 'mapping.generator'`

- [ ] **Step 3: Write minimal implementation**

3a. 创建 `mapping/generator/transformer.py`：

```python
"""预置积木块：TransformerBlock（pre-norm）。"""

import torch
import torch.nn as nn

from mapping.generator.block import Block
from mapping.generator.mlp import MLP


class TransformerBlock(Block):
    """pre-norm Transformer 块。

    结构：x + attn(norm1(x))，再 x + ffn(norm2(x))。
    输入形状 (B, L, D)，输出形状不变。内部全部参数固定。

    Args:
        dim: 特征维度 D
        num_heads: 注意力头数（须整除 dim）
        mlp_ratio: FFN 隐藏层倍数（默认 4.0）
        dropout: 注意力 dropout（默认 0.0）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP([dim, int(dim * mlp_ratio), dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))
```

3b. `mapping/generator/__init__.py` 更新为：

```python
from mapping.generator.block import Block
from mapping.generator.linear import Linear
from mapping.generator.conv import Conv1d, Conv2d
from mapping.generator.lrd import LRDLayer
from mapping.generator.mlp import MLP
from mapping.generator.resblock import ConvResBlock, LinearResBlock
from mapping.generator.transformer import TransformerBlock

__all__ = [
    'Block', 'Linear', 'Conv1d', 'Conv2d', 'LRDLayer',
    'MLP', 'LinearResBlock', 'ConvResBlock', 'TransformerBlock',
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generator_transformer.py -v`
Expected: 6 passed

Run: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest`
Expected: 全量通过（139 + 6 = 145）

- [ ] **Step 5: Commit**

```bash
git add mapping/generator/transformer.py mapping/generator/__init__.py tests/test_generator_transformer.py
git commit -m "feat: add mapping.generator.TransformerBlock prebuilt block"
```

---

## Self-Review 记录

- **Spec 覆盖**：§2.3 要求 `MLP` / `ResBlock`（linear 版与 conv 版）/ `TransformerBlock`（pre-norm，内部参数固定）→ Task 1/2/3 全覆盖；「全部基于 Block，与用户自定义写法同一套机制」→ 三个模块均直接继承 `Block` 并组合叶子块。
- **命名说明**：spec 原文为 `generator.ResBlock`（linear 版与 conv 版），为避免单类承担两种输入形状，拆为 `LinearResBlock` / `ConvResBlock` 两个具名类——与阶段 1 测试 `test_residual_block_with_generator_subblocks` 中用户的 conv 残差写法一致。
- **类型一致性**：`MLP(sizes, act)`、`LinearResBlock(dim)`、`ConvResBlock(channels, kernel_size=3)`、`TransformerBlock(dim, num_heads, mlp_ratio=4.0, dropout=0.0)` 在测试与实现中一致；`MLP` 暴露 `layers: nn.ModuleList`，Task 3 测试通过 `ffn.layers[-1]` 依赖此接口。
- **初始化语义**：预置块默认不重载 `init_weights`（no-op），叶子 Linear/Conv2d 在自身构造时已完成 kaiming 初始化；`test_residual_property` 验证用户重载 `init_weights` 在所有子模块构造**之后**执行。
