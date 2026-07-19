import pytest
import torch

from mapping.base import Generator
from mapping.layers import Conv2d, Linear
from mapping.sequential import Sequential


# --- 测试用的 Generator ---
class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[: self.w_size].reshape(self.w_shape)
        b = h[self.w_size :].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestSequential:
    def test_slvt_forward(self, device):
        """SLVT Sequential 基本前向。"""
        # Conv2d(1,20,5)->(2,20,24,24), Conv2d(20,32,5)->(2,32,20,20),
        # Flatten->(2,12800), Linear(12800,10)->(2,10)
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            torch.nn.Flatten(1),
            Linear(12800, 10),
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape[0] == 2

    def test_mixed_param_and_nonparam_layers(self, device):
        """可混装非参数层（ReLU, MaxPool2d, Flatten）。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(20, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(512, 10),
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_rejects_layers_with_own_generator(self, device):
        """传入自带 generator 的层时报错。"""
        with pytest.raises(ValueError, match='自带 generator'):
            Sequential(
                Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32),
                generator_cls=SimpleGen,
                z_dim=64,
            )

    def test_gradient_flows(self, device):
        """梯度通过共享 generator 流向 z。"""
        # Conv2d(1,20,5)->(2,20,24,24), Flatten->(2,11520), Linear(11520,10)->(2,10)
        net = Sequential(
            Conv2d(1, 20, 5),
            torch.nn.Flatten(1),
            Linear(11520, 10),
            generator_cls=SimpleGen,
            z_dim=64,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        loss = y.sum()
        loss.backward()
        assert net.generator.z.grad is not None

    def test_generator_kwargs_passthrough(self, device):
        """**generator_kwargs 透传给 Generator。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, my_param=42):
                super().__init__(param_spec, z_dim=z_dim)
                self.my_param = my_param
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape) if self.b_size > 0 else None
                return w, b

        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            Linear(512, 10),
            generator_cls=KwargsGen,
            z_dim=64,
            my_param=99,
        ).to(device)
        assert net.generator.my_param == 99
