import torch
import torch.nn as nn
from mapping.generator import Block, ConvResBlock, LinearResBlock


class TestLinearResBlock:
    def test_is_block_and_frozen(self, device):
        block = LinearResBlock(16).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        assert block(x).shape == (2, 16)

    def test_forward_matches_manual(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        expected = x + block.fc2(block.act(block.fc1(x)))
        assert torch.equal(block(x), expected)

    def test_residual_property(self, device):
        """fc2 输出为 0 时，输出严格等于输入（跳连恒等）。
        同时验证 init_weights 钩子在子块构造之后执行。"""

        class ZeroResBlock(LinearResBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.fc2.weight)
                nn.init.zeros_(self.fc2.bias)

        block = ZeroResBlock(16).to(device)
        x = torch.randn(2, 16, device=device)
        assert torch.equal(block(x), x)

    def test_gradient_flows_to_input(self, device):
        block = LinearResBlock(16).to(device)
        x = torch.randn(2, 16, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


class TestConvResBlock:
    def test_is_block_and_frozen(self, device):
        block = ConvResBlock(8).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        """通道与空间尺寸不变。"""
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert block(x).shape == (2, 8, 10, 10)

    def test_custom_kernel_size(self, device):
        block = ConvResBlock(8, kernel_size=5).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert block(x).shape == (2, 8, 10, 10)

    def test_forward_matches_manual(self, device):
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        expected = x + block.conv2(block.act(block.conv1(x)))
        assert torch.equal(block(x), expected)

    def test_residual_property(self, device):
        """conv2 输出为 0 时，输出严格等于输入。"""

        class ZeroConvResBlock(ConvResBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.conv2.weight)
                nn.init.zeros_(self.conv2.bias)

        block = ZeroConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device)
        assert torch.equal(block(x), x)

    def test_gradient_flows_to_input(self, device):
        block = ConvResBlock(8).to(device)
        x = torch.randn(2, 8, 10, 10, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
