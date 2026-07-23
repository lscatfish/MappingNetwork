"""CNN1 (AlexNet 风格) — 新框架重写。

结构：Conv2d(1,48,5) -> AvgPool -> Conv2d(48,128,5) -> AvgPool -> FC(2048,186) -> FC(186,10)
约 537,960 参数。
"""

import torch.nn as nn

from mapping import Conv2d, Linear, Sequential


def cnn1_slvt(generator_cls, **gen_kwargs):
    return Sequential(
        Conv2d(1, 48, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        Conv2d(48, 128, 5),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(1),
        Linear(2048, 186),
        nn.ReLU(),
        Linear(186, 10),
        generator_cls=generator_cls,
        **gen_kwargs,
    )


def cnn1_lwt(generator_cls, **gen_kwargs):
    class CNN1LWT(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(1, 48, 5, generator_cls=generator_cls, **gen_kwargs)
            self.conv2 = Conv2d(48, 128, 5, generator_cls=generator_cls, **gen_kwargs)
            self.fc1 = Linear(2048, 186, generator_cls=generator_cls, **gen_kwargs)
            self.fc2 = Linear(186, 10, generator_cls=generator_cls, **gen_kwargs)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = x.flatten(1)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    return CNN1LWT()


def cnn1_baseline():
    class CNN1Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 48, 5)
            self.conv2 = nn.Conv2d(48, 128, 5)
            self.fc1 = nn.Linear(2048, 186)
            self.fc2 = nn.Linear(186, 10)

        def forward(self, x):
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv1(x)), 2)
            x = nn.functional.avg_pool2d(nn.functional.relu(self.conv2(x)), 2)
            x = x.flatten(1)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    return CNN1Baseline()
