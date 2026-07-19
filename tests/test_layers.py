import torch
import pytest
import torch.nn.functional as F
from mapping.base import Generator, MappingLayer
from mapping.layers import Conv2d, Linear


# --- 测试用的 Generator ---
class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestConv2d:
    def test_param_spec_auto_deduced(self, device):
        """Conv2d 自动推导 param_spec。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (20, 1, 5, 5)
        assert layer.param_spec['bias'] == (20,)

    def test_param_spec_no_bias(self, device):
        """bias=False 时 param_spec 不含 bias。"""
        layer = Conv2d(1, 20, 5, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        """forward 输出形状正确。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = layer(x)
        assert y.shape == (2, 20, 24, 24)

    def test_forward_with_params(self, device):
        """forward_with_params 接收外部参数。"""
        layer = Conv2d(1, 20, 5).to(device)  # 纯形状层
        x = torch.randn(2, 1, 28, 28, device=device)
        w = torch.randn(20, 1, 5, 5, device=device)
        b = torch.randn(20, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv2d(x, w, b)
        assert torch.allclose(y, expected)

    def test_flat_params_auto_reshape(self, device):
        """flat 参数自动 reshape。"""
        layer = Conv2d(1, 20, 5).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        w_flat = torch.randn(500, device=device)   # 20*1*5*5
        b_flat = torch.randn(20, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 20, 24, 24)

    def test_stride_padding(self, device):
        """stride 和 padding 参数生效。"""
        layer = Conv2d(3, 16, 3, stride=2, padding=1).to(device)
        x = torch.randn(2, 3, 10, 10, device=device)
        w = torch.randn(16, 3, 3, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv2d(x, w, b, stride=2, padding=1)
        assert torch.allclose(y, expected)

    def test_generator_kwargs_passthrough(self, device):
        """**generator_kwargs 透传给 Generator。"""

        class KwargsGen(Generator):
            def __init__(self, param_spec, z_dim, my_param=42):
                super().__init__(param_spec, z_dim=z_dim)
                self.my_param = my_param
                self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

            def forward(self):
                h = self.head(self.z)
                w = h[:self.w_size].reshape(self.w_shape)
                b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
                return w, b

        layer = Conv2d(1, 20, 5, generator_cls=KwargsGen, z_dim=32, my_param=99).to(device)
        assert layer.generator.my_param == 99

    def test_pure_shape_layer_no_generator(self, device):
        """不传 generator_cls 时，层没有 generator 属性。"""
        layer = Conv2d(1, 20, 5).to(device)
        assert not hasattr(layer, 'generator')

    def test_gradient_flows_through_generator(self, device):
        """梯度通过 generator 流向 z。"""
        layer = Conv2d(1, 20, 5, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.generator.z.grad is not None


class TestLinear:
    def test_param_spec_auto_deduced(self, device):
        """Linear 自动推导 param_spec。"""
        layer = Linear(512, 176, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (176, 512)
        assert layer.param_spec['bias'] == (176,)

    def test_forward_output_shape(self, device):
        """forward 输出形状正确。"""
        layer = Linear(512, 176, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 512, device=device)
        y = layer(x)
        assert y.shape == (2, 176)

    def test_forward_with_params(self, device):
        """forward_with_params 接收外部参数。"""
        layer = Linear(512, 176).to(device)
        x = torch.randn(2, 512, device=device)
        w = torch.randn(176, 512, device=device)
        b = torch.randn(176, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.linear(x, w, b)
        assert torch.allclose(y, expected)

    def test_no_bias(self, device):
        """bias=False 时，forward_with_params 容错。"""
        layer = Linear(512, 176, bias=False).to(device)
        x = torch.randn(2, 512, device=device)
        w = torch.randn(176, 512, device=device)
        y = layer.forward_with_params(x, w, None)
        expected = F.linear(x, w)
        assert torch.allclose(y, expected)
