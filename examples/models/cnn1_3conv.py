"""CNN1_3Conv (三卷积实验版) — 新框架重写。

结构：Conv2d(1,16,5) -> AvgPool -> Conv2d(16,32,5) -> AvgPool -> Conv2d(32,64,3) -> AvgPool -> FC(64,10)
约 32,394 参数。
"""

import torch.nn as nn

from mapping import Conv2d, Linear, Sequential


def cnn1_3conv_slvt(generator_cls, **gen_kwargs):
    return Sequential(
        Conv2d(1, 16, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Conv2d(16, 32, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(1),
        Linear(64, 10),
        generator_cls=generator_cls,
        **gen_kwargs,
    )


def cnn1_3conv_lwt(generator_cls, **gen_kwargs):
    class CNN1_3ConvLWT(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(1, 16, 5, generator_cls=generator_cls, **gen_kwargs)
            self.conv2 = Conv2d(16, 32, 5, generator_cls=generator_cls, **gen_kwargs)
            self.conv3 = Conv2d(32, 64, 3, generator_cls=generator_cls, **gen_kwargs)
            self.fc1 = Linear(64, 10, generator_cls=generator_cls, **gen_kwargs)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv3(x)), 2)
            x = x.flatten(1)
            return self.fc1(x)

    return CNN1_3ConvLWT()


def cnn1_3conv_baseline():
    class CNN1_3ConvBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 5)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.conv3 = nn.Conv2d(32, 64, 3)
            self.fc1 = nn.Linear(64, 10)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv3(x)), 2)
            x = x.flatten(1)
            return self.fc1(x)

    return CNN1_3ConvBaseline()
