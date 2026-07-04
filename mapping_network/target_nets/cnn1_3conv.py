import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TargetNet

class CNN1_3Conv(TargetNet):
    """AlexNet 风格三卷积版（实验性）。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)     # 416
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)    # 12,832
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)    # 18,496
        self.pool3 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(64, 10)                     # 650
        self._build_param_slices()

    def _functional_forward(self, x, params):
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias']))
        x = self.pool1(x)
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias']))
        x = self.pool2(x)
        x = F.relu(F.conv2d(x, params['conv3.weight'], params['conv3.bias']))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
