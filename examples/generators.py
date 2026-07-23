"""示例用 Generator 实现。"""

import torch.nn as nn

from mapping import Generator
from mapping.generator import MLP as MLPBlock


class MLPGenerator(Generator):
    """基于 MLP 积木块的参数生成器。

    z -> MLP body -> w_head / b_head -> (weight, bias)
    """

    def __init__(self, param_spec, z_dim=64, hidden_dim=256, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.body = nn.Sequential(
            MLPBlock([z_dim, hidden_dim]),
            nn.ReLU(),
        )
        self.w_head = nn.Linear(hidden_dim, self.w_size)
        self.b_head = (
            nn.Linear(hidden_dim, self.b_size) if self.b_size > 0 else None
        )

    def forward(self):
        h = self.body(self.z)
        w = self.w_head(h).reshape(self.w_shape)
        b = self.b_head(h).reshape(self.b_shape) if self.b_head is not None else None
        return w, b
