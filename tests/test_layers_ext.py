import torch
import torch.nn.functional as F

from mapping.base import Generator
from mapping.layers import BatchNorm1d, BatchNorm2d, Conv1d, ConvTranspose2d


class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestConv1d:
    def test_param_spec_auto_deduced(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16, 4, 3)
        assert layer.param_spec['bias'] == (16,)

    def test_param_spec_no_bias(self, device):
        layer = Conv1d(4, 16, 3, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        assert y.shape == (2, 16, 98)

    def test_forward_with_params(self, device):
        layer = Conv1d(4, 16, 3).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w = torch.randn(16, 4, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv1d(x, w, b)
        assert torch.allclose(y, expected)

    def test_flat_params_auto_reshape(self, device):
        layer = Conv1d(4, 16, 3).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w_flat = torch.randn(192, device=device)
        b_flat = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 16, 98)

    def test_stride_padding(self, device):
        layer = Conv1d(4, 16, 3, stride=2, padding=1).to(device)
        x = torch.randn(2, 4, 100, device=device)
        w = torch.randn(16, 4, 3, device=device)
        b = torch.randn(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv1d(x, w, b, stride=2, padding=1)
        assert torch.allclose(y, expected)

    def test_gradient_flows(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None


class TestConvTranspose2d:
    def test_param_spec_auto_deduced(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16, 8, 3, 3)
        assert layer.param_spec['bias'] == (8,)

    def test_param_spec_no_bias(self, device):
        layer = ConvTranspose2d(16, 8, 3, bias=False, generator_cls=SimpleGen, z_dim=32).to(device)
        assert 'bias' not in layer.param_spec

    def test_forward_output_shape(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 9, 9)

    def test_forward_with_params(self, device):
        layer = ConvTranspose2d(16, 8, 3).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w = torch.randn(16, 8, 3, 3, device=device)
        b = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv_transpose2d(x, w, b)
        assert torch.allclose(y, expected)

    def test_stride_padding_output_padding(self, device):
        layer = ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w = torch.randn(16, 8, 3, 3, device=device)
        b = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)
        assert torch.allclose(y, expected)
        assert y.shape == (2, 8, 14, 14)

    def test_flat_params_auto_reshape(self, device):
        layer = ConvTranspose2d(16, 8, 3).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        w_flat = torch.randn(1152, device=device)
        b_flat = torch.randn(8, device=device)
        y = layer.forward_with_params(x, w_flat, b_flat)
        assert y.shape == (2, 8, 9, 9)

    def test_gradient_flows(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None


class TestBatchNorm2d:
    def test_param_spec(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (16,)
        assert layer.param_spec['bias'] == (16,)

    def test_buffers_registered(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        buffers = dict(layer.named_buffers())
        assert 'running_mean' in buffers
        assert 'running_var' in buffers
        assert 'num_batches_tracked' in buffers
        assert layer.running_mean.shape == (16,)
        assert layer.running_var.shape == (16,)

    def test_forward_train_mode(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)

    def test_forward_eval_mode(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.eval()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)

    def test_forward_with_params(self, device):
        layer = BatchNorm2d(16).to(device)
        layer.eval()
        x = torch.randn(4, 16, 8, 8, device=device)
        w = torch.ones(16, device=device)
        b = torch.zeros(16, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.batch_norm(
            x, layer.running_mean, layer.running_var, w, b, False, 0.1, 1e-5
        )
        assert torch.allclose(y, expected)

    def test_running_stats_updated_in_train(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        _ = layer(x)
        assert not torch.allclose(layer.running_mean, torch.zeros(16, device=device))

    def test_gradient_flows(self, device):
        layer = BatchNorm2d(16, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_custom_eps_momentum(self, device):
        layer = BatchNorm2d(16, eps=1e-3, momentum=0.2,
                            generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.eps == 1e-3
        assert layer.momentum == 0.2


class TestBatchNorm1d:
    def test_param_spec(self, device):
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert layer.param_spec['weight'] == (32,)
        assert layer.param_spec['bias'] == (32,)

    def test_forward_2d_input(self, device):
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(8, 32, device=device)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_forward_3d_input(self, device):
        layer = BatchNorm1d(32, generator_cls=SimpleGen, z_dim=32).to(device)
        layer.train()
        x = torch.randn(4, 32, 100, device=device)
        y = layer(x)
        assert y.shape == (4, 32, 100)

    def test_forward_with_params(self, device):
        layer = BatchNorm1d(32).to(device)
        layer.eval()
        x = torch.randn(8, 32, device=device)
        w = torch.ones(32, device=device)
        b = torch.zeros(32, device=device)
        y = layer.forward_with_params(x, w, b)
        expected = F.batch_norm(
            x, layer.running_mean, layer.running_var, w, b, False, 0.1, 1e-5
        )
        assert torch.allclose(y, expected)
