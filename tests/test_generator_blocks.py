import torch
import pytest
from mapping.generator.linear import Linear


class TestGeneratorLinear:
    def test_init_aligns_torch(self, device):
        """init 签名对齐 torch.nn.Linear。"""
        layer = Linear(10, 20).to(device)
        assert layer.weight.shape == (20, 10)
        assert layer.bias.shape == (20,)

    def test_params_are_frozen(self, device):
        """内部参数 requires_grad=False。"""
        layer = Linear(10, 20).to(device)
        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad

    def test_no_bias(self, device):
        """bias=False 时 bias 为 None。"""
        layer = Linear(10, 20, bias=False).to(device)
        assert layer.bias is None

    def test_forward_matches_torch(self, device):
        """forward 行为与 F.linear 一致。"""
        layer = Linear(10, 20).to(device)
        x = torch.randn(4, 10, device=device)
        y = layer(x)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.allclose(y, expected)

    def test_init_weights_called_on_construction(self, device):
        """构造时自动调用 init_weights。"""
        layer = Linear(10, 20).to(device)
        # 权重非零且非全等（已初始化）
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_custom_init_weights(self, device):
        """用户可重载 init_weights 自定义初始化。"""

        class CustomLinear(Linear):
            def init_weights(self):
                torch.nn.init.ones_(self.weight)
                if self.bias is not None:
                    torch.nn.init.zeros_(self.bias)

        layer = CustomLinear(10, 20).to(device)
        assert torch.allclose(layer.weight, torch.ones_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_forward_preserves_gradient(self, device):
        """forward 输出可反向传播（子块参数虽固定，但输入可梯度）。"""
        layer = Linear(10, 20).to(device)
        x = torch.randn(4, 10, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
