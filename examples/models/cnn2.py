"""CNN2 (LeNet 风格) — 新框架重写。

结构：Conv2d(1,20,5) -> AvgPool -> Conv2d(20,32,5) -> AvgPool -> FC(512,176) -> FC(176,10)
约 108,610 参数。
"""

import torch.nn as nn

from mapping import Conv2d, Linear, Sequential


def cnn2_slvt(generator_cls, **gen_kwargs):
    """SLVT 模式：Sequential 共享 generator。"""
    return Sequential(
        Conv2d(1, 20, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Conv2d(20, 32, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(1),
        Linear(512, 176),
        nn.ReLU(),
        Linear(176, 10),
        generator_cls=generator_cls,
        **gen_kwargs,
    )


def cnn2_lwt(generator_cls, **gen_kwargs):
    """LWT 模式：逐层 generator。"""

    class CNN2LWT(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(1, 20, 5, generator_cls=generator_cls, **gen_kwargs)
            self.conv2 = Conv2d(20, 32, 5, generator_cls=generator_cls, **gen_kwargs)
            self.fc1 = Linear(512, 176, generator_cls=generator_cls, **gen_kwargs)
            self.fc2 = Linear(176, 10, generator_cls=generator_cls, **gen_kwargs)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = x.flatten(1)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    return CNN2LWT()


def cnn2_baseline():
    """Baseline：纯 torch 实现。"""

    class CNN2Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 32, 5)
            self.fc1 = nn.Linear(512, 176)
            self.fc2 = nn.Linear(176, 10)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = x.flatten(1)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    return CNN2Baseline()
