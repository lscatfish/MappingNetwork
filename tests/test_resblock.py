import torch

from mapping.base import Generator, MappingLayer, _prod
from mapping.resblock import ResBlock


class SimpleGen(Generator):
    def __init__(self, param_spec, z_dim=32, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestResBlockLWT:
    """LWT 模式：内部各层自带 generator。"""

    def test_is_mapping_layer(self, device):
        block = ResBlock(16, 16, generator_cls=SimpleGen, z_dim=32).to(device)
        assert isinstance(block, MappingLayer)

    def test_forward_same_channels(self, device):
        block = ResBlock(16, 16, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 16, 8, 8)

    def test_forward_channel_change_enables_shortcut(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert block.use_shortcut
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 32, 8, 8)

    def test_forward_stride_change_enables_shortcut(self, device):
        block = ResBlock(16, 16, stride=2, generator_cls=SimpleGen, z_dim=32).to(device)
        assert block.use_shortcut
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 16, 4, 4)

    def test_internal_layers_have_generators(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        assert hasattr(block.conv1, 'generator')
        assert hasattr(block.conv2, 'generator')
        assert hasattr(block.shortcut, 'generator')

    def test_gradient_flows_to_all_z(self, device):
        block = ResBlock(16, 32, generator_cls=SimpleGen, z_dim=32).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        y.sum().backward()
        assert block.conv1.generator.z.grad is not None
        assert block.conv2.generator.z.grad is not None
        assert block.shortcut.generator.z.grad is not None


class TestResBlockSLVT:
    """SLVT 模式：纯形状层，聚合 param_spec。"""

    def test_no_generator_on_block(self, device):
        block = ResBlock(16, 16).to(device)
        assert not hasattr(block, 'generator')

    def test_aggregated_param_spec_same_channels(self, device):
        block = ResBlock(16, 16).to(device)
        conv1_w = 16 * 16 * 3 * 3
        conv2_w = 16 * 16 * 3 * 3
        total_w = conv1_w + conv2_w
        total_b = 16 + 16
        assert block.param_spec == {'weight': (total_w,), 'bias': (total_b,)}

    def test_aggregated_param_spec_with_shortcut(self, device):
        block = ResBlock(16, 32).to(device)
        conv1_w = 32 * 16 * 3 * 3
        conv2_w = 32 * 32 * 3 * 3
        shortcut_w = 32 * 16 * 1 * 1
        total_w = conv1_w + conv2_w + shortcut_w
        total_b = 32 + 32 + 32
        assert block.param_spec == {'weight': (total_w,), 'bias': (total_b,)}

    def test_forward_with_params(self, device):
        block = ResBlock(16, 16).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.randn(total_w, device=device)
        b = torch.randn(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert y.shape == (2, 16, 8, 8)

    def test_forward_with_params_shortcut(self, device):
        block = ResBlock(16, 32, stride=2).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.randn(total_w, device=device)
        b = torch.randn(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert y.shape == (2, 32, 4, 4)

    def test_no_bias(self, device):
        block = ResBlock(16, 16, bias=False).to(device)
        assert 'bias' not in block.param_spec
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        w = torch.randn(total_w, device=device)
        y = block.forward_with_params(x, w, None)
        assert y.shape == (2, 16, 8, 8)

    def test_residual_property(self, device):
        """零权重时输出等于输入（同通道无 stride）。"""
        block = ResBlock(16, 16).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        total_w = _prod(block.param_spec['weight'])
        total_b = _prod(block.param_spec['bias'])
        w = torch.zeros(total_w, device=device)
        b = torch.zeros(total_b, device=device)
        y = block.forward_with_params(x, w, b)
        assert torch.allclose(y, x, atol=1e-6)
