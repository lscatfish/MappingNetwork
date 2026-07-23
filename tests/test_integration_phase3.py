import torch

from mapping import (
    BatchNorm2d,
    Conv1d,
    Conv2d,
    ConvTranspose2d,
    Generator,
    Linear,
    ResBlock,
    Sequential,
)
from mapping.generator import Linear as GenLinear


class MyGen(Generator):
    def __init__(self, param_spec, z_dim=64, hidden_dim=128):
        super().__init__(param_spec, z_dim=z_dim)
        self.body = torch.nn.Sequential(
            GenLinear(z_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.w_head = torch.nn.Linear(hidden_dim, self.w_size)
        self.b_head = (
            torch.nn.Linear(hidden_dim, self.b_size) if self.b_size > 0 else None
        )

    def forward(self):
        h = self.body(self.z)
        w = self.w_head(h).reshape(self.w_shape)
        b = self.b_head(h).reshape(self.b_shape) if self.b_head is not None else None
        return w, b


class TestSequentialWithResBlock:
    """ResBlock 作为纯形状层放入 Sequential。"""

    def test_slvt_with_resblock(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            ResBlock(16, 16),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(16 * 14 * 14, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_slvt_with_resblock_channel_change(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            ResBlock(16, 32, stride=2),
            torch.nn.Flatten(1),
            Linear(32 * 14 * 14, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

    def test_gradient_flows_through_resblock(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            ResBlock(16, 16),
            torch.nn.Flatten(1),
            Linear(16 * 28 * 28, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        y.sum().backward()
        assert net.generator.z.grad is not None


class TestLWTWithNewLayers:
    """LWT 模式使用新层类型。"""

    def test_lwt_conv1d(self, device):
        layer = Conv1d(4, 16, 3, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 4, 100, device=device)
        y = layer(x)
        assert y.shape == (2, 16, 98)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_conv_transpose2d(self, device):
        layer = ConvTranspose2d(16, 8, 3, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 16, 7, 7, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 9, 9)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_batchnorm2d(self, device):
        layer = BatchNorm2d(16, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        layer.train()
        x = torch.randn(4, 16, 8, 8, device=device)
        y = layer(x)
        assert y.shape == (4, 16, 8, 8)
        y.sum().backward()
        assert layer.generator.z.grad is not None

    def test_lwt_resblock(self, device):
        block = ResBlock(16, 32, generator_cls=MyGen, z_dim=64, hidden_dim=64).to(device)
        x = torch.randn(2, 16, 8, 8, device=device)
        y = block(x)
        assert y.shape == (2, 32, 8, 8)
        y.sum().backward()
        assert block.conv1.generator.z.grad is not None


class TestSequentialWithBatchNorm:
    """BatchNorm 在 Sequential 中的行为。"""

    def test_slvt_with_batchnorm(self, device):
        net = Sequential(
            Conv2d(1, 16, 3, padding=1),
            BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            Linear(16 * 28 * 28, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)
        net.train()
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)
