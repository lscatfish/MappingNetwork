import torch
import torch.nn.functional as F
from mapping import Generator, Conv2d, Linear, Sequential
from mapping.generator import Linear as GenLinear, Conv2d as GenConv2d


# --- 用户自定义 Generator（模拟真实使用场景）---
class MyGen(Generator):
    """用户自定义 Generator：使用 generator 子块组合。"""

    def __init__(self, param_spec, z_dim=64, hidden_dim=128):
        super().__init__(param_spec, z_dim=z_dim)
        self.body = torch.nn.Sequential(
            GenLinear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            GenLinear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
        )
        self.w_head = torch.nn.Linear(hidden_dim * 2, self.w_size)
        self.b_head = (
            torch.nn.Linear(hidden_dim * 2, self.b_size) if self.b_size > 0 else None
        )

    def forward(self):
        h = self.body(self.z)
        w = self.w_head(h).reshape(self.w_shape)
        b = self.b_head(h).reshape(self.b_shape) if self.b_head is not None else None
        return w, b


class TestIntegrationLWT:
    """LWT 模式集成测试：逐层 generator，直接堆叠。"""

    def test_lwt_forward(self, device):
        """LWT 完整前向：conv1 -> pool -> conv2 -> pool -> fc1 -> fc2。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(
                    1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )
                self.conv2 = Conv2d(
                    20, 32, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )
                self.fc1 = Linear(
                    512, 176, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )
                self.fc2 = Linear(
                    176, 10, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = F.relu(self.fc1(x.flatten(1)))
                return self.fc2(x)

        net = LWTNet().to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)

    def test_lwt_gradient_flows(self, device):
        """LWT 各层 generator 的 z 独立训练。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(
                    1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )

            def forward(self, x):
                return F.relu(self.conv1(x))

        net = LWTNet().to(device)
        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        loss = y.sum()
        loss.backward()

        assert net.conv1.generator.z.grad is not None
        assert not torch.allclose(
            net.conv1.generator.z.grad, torch.zeros_like(net.conv1.generator.z.grad)
        )

    def test_lwt_each_layer_has_own_z(self, device):
        """LWT 每层有独立的 z。"""

        class LWTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(
                    1, 20, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )
                self.conv2 = Conv2d(
                    20, 32, 5, generator_cls=MyGen, z_dim=64, hidden_dim=128
                )

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                return x

        net = LWTNet().to(device)
        # 两层 z 独立
        assert not torch.equal(
            net.conv1.generator.z.data, net.conv2.generator.z.data
        )


class TestIntegrationSLVT:
    """SLVT 模式集成测试：共享 generator。"""

    def test_slvt_full_forward(self, device):
        """SLVT 完整前向：Sequential 共享 generator。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            Conv2d(20, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            Linear(512, 176),
            torch.nn.ReLU(),
            Linear(176, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=256,
        ).to(device)

        x = torch.randn(4, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (4, 10)

    def test_slvt_single_z(self, device):
        """SLVT 只有一个 z。"""
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            torch.nn.Flatten(1),
            Linear(12800, 10),
            generator_cls=MyGen,
            z_dim=64,
            hidden_dim=128,
        ).to(device)

        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)

        assert net.generator.z.shape == (64,)
        assert net.generator.z.requires_grad


class TestFlatKerasStyle:
    """用户像 torch 一样写代码。"""

    def test_concise_syntax(self, device):
        """
        用户可以直接用简洁的语法构建网络。
        验证 init 中不出现嵌套的 generator_kwargs dict。
        """
        # 这是设计文档中期望的最终用户语法
        net = Sequential(
            Conv2d(1, 20, 5),
            Conv2d(20, 32, 5),
            torch.nn.Flatten(1),
            Linear(12800, 10),
            generator_cls=MyGen,
            z_dim=64,           # 直接透传，不是 generator_kwargs={'z_dim': 64}
            hidden_dim=128,     # 直接透传
        ).to(device)

        x = torch.randn(2, 1, 28, 28, device=device)
        y = net(x)
        assert y.shape == (2, 10)
