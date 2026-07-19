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
