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
