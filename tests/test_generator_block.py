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
