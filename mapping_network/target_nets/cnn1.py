import torch.nn as nn
import torch.nn.functional as F

from .base import TargetNet


class CNN1(TargetNet):
    """AlexNet 风格，~537,960 参数。"""

    def __init__(self, lrd_config=None):
        super().__init__(lrd_config)
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5)  # 1,200 params
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5)  # 153,728 params
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(2048, 186)  # 381,114 params
        self.fc2 = nn.Linear(186, 10)  # 1,870 params
        self._build_param_slices()

    def _functional_forward(self, x, params):
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias']))
        x = self.pool1(x)
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias']))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
