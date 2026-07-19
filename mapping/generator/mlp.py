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
