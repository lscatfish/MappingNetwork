import pytest
import torch

from mapping.base import Generator


class TestGenerator:
    def test_z_is_trainable_parameter(self, device):
        """z 是 nn.Parameter 且 requires_grad=True。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape)
                return w, b

        gen = SimpleGen({'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64).to(device)
        assert isinstance(gen.z, torch.nn.Parameter)
        assert gen.z.requires_grad

    def test_auto_derived_attrs(self, device):
        """基类自动派生 w_shape/b_shape/w_size/b_size。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape)
                return w, b

        gen = SimpleGen({'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64).to(device)
        assert gen.w_shape == (20, 1, 5, 5)
        assert gen.b_shape == (20,)
        assert gen.w_size == 500  # 20*1*5*5
        assert gen.b_size == 20

    def test_no_bias(self, device):
        """bias=False 时 b_shape=None, b_size=0。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size)

            def forward(self):
                h = self.head(self.z)
                return h.reshape(self.w_shape), None

        gen = SimpleGen({'weight': (10, 5)}, z_dim=32).to(device)
        assert gen.b_shape is None
        assert gen.b_size == 0

    def test_forward_returns_tuple(self, device):
        """forward 返回 (weight, bias) tuple。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape)
                return w, b

        gen = SimpleGen({'weight': (20, 1, 5, 5), 'bias': (20,)}, z_dim=64).to(device)
        w, b = gen()
        assert w.shape == (20, 1, 5, 5)
        assert b.shape == (20,)

    def test_forward_is_abstract(self):
        """Generator 不可直接实例化（forward 是抽象的）。"""

        with pytest.raises(TypeError):
            Generator({'weight': (10,), 'bias': (10,)}, z_dim=32)

    def test_kwargs_passthrough(self, device):
        """**kwargs 透传给子类 __init__。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, hidden_dim=128, **kwargs):
                super().__init__(param_spec, z_dim=z_dim, **kwargs)
                self.hidden_dim = hidden_dim
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape)
                return w, b

        gen = KwargsGen({'weight': (20,), 'bias': (20,)}, z_dim=64, hidden_dim=256).to(device)
        assert gen.hidden_dim == 256

    def test_z_gradient_flows(self, device):
        """z 的梯度正常流动。"""

        class SimpleGen(Generator):
            def __init__(self, param_spec, z_dim):
                super().__init__(param_spec, z_dim=z_dim)
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[: self.w_size].reshape(self.w_shape)
                b = h[self.w_size :].reshape(self.b_shape)
                return w, b

        gen = SimpleGen({'weight': (10, 5), 'bias': (10,)}, z_dim=32).to(device)
        w, b = gen()
        loss = w.sum() + b.sum()
        loss.backward()
        assert gen.z.grad is not None
        assert not torch.allclose(gen.z.grad, torch.zeros_like(gen.z.grad))
