import pytest
import torch
import torch.nn as nn
from mapping.generator import Block, Linear, MLP


class TestMLP:
    def test_is_block_and_frozen(self, device):
        """MLP 是 Block 子类，全部参数自动冻结。"""
        mlp = MLP([8, 16, 32]).to(device)
        assert isinstance(mlp, Block)
        params = list(mlp.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_structure(self, device):
        """len(sizes)-1 个 Linear，激活夹在中间，最后一个模块是 Linear。"""
        mlp = MLP([8, 16, 32, 4]).to(device)
        linears = [m for m in mlp.layers if isinstance(m, Linear)]
        acts = [m for m in mlp.layers if isinstance(m, nn.ReLU)]
        assert len(linears) == 3
        assert len(acts) == 2
        assert isinstance(mlp.layers[-1], Linear)
        assert mlp.layers[0].in_features == 8
        assert mlp.layers[-1].out_features == 4

    def test_forward_shape(self, device):
        mlp = MLP([8, 16, 32]).to(device)
        x = torch.randn(2, 8, device=device)
        assert mlp(x).shape == (2, 32)

    def test_forward_matches_manual(self, device):
        """输出等于逐层手动计算。"""
        mlp = MLP([8, 16, 32]).to(device)
        x = torch.randn(2, 8, device=device)
        h = x
        for layer in mlp.layers:
            h = layer(h)
        assert torch.equal(mlp(x), h)

    def test_sizes_too_short_raises(self, device):
        with pytest.raises(ValueError):
            MLP([8])

    def test_custom_activation(self, device):
        mlp = MLP([8, 16, 8], act=nn.GELU).to(device)
        acts = [m for m in mlp.layers if isinstance(m, nn.GELU)]
        assert len(acts) == 1

    def test_gradient_flows_to_input(self, device):
        """子块参数固定，但输入可梯度。"""
        mlp = MLP([8, 16, 8]).to(device)
        x = torch.randn(2, 8, device=device, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None
