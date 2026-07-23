import torch
import torch.nn as nn
from mapping.generator import Block, MLP, TransformerBlock


class TestTransformerBlock:
    def test_is_block_and_frozen(self, device):
        """TransformerBlock 是 Block 子类，全部参数（含 attn/LayerNorm）冻结。"""
        block = TransformerBlock(16, 4).to(device)
        assert isinstance(block, Block)
        params = list(block.parameters())
        assert len(params) > 0
        for p in params:
            assert not p.requires_grad

    def test_forward_shape(self, device):
        block = TransformerBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device)  # (B, L, D)
        assert block(x).shape == (2, 5, 16)

    def test_deterministic_without_dropout(self, device):
        """dropout=0 时两次前向结果一致。"""
        block = TransformerBlock(16, 4, dropout=0.0).to(device)
        x = torch.randn(2, 5, 16, device=device)
        assert torch.equal(block(x), block(x))

    def test_residual_property(self, device):
        """attn 输出投影与 ffn 末层置零时，输出等于输入（跳连恒等）。
        同时验证 init_weights 钩子在子模块构造之后执行。"""

        class ZeroBlock(TransformerBlock):
            def init_weights(self) -> None:
                nn.init.zeros_(self.attn.out_proj.weight)
                nn.init.zeros_(self.attn.out_proj.bias)
                last = self.ffn.layers[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

        block = ZeroBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device)
        assert torch.allclose(block(x), x)

    def test_mlp_ratio(self, device):
        """FFN 隐藏层维度 = int(dim * mlp_ratio)。"""
        block = TransformerBlock(16, 4, mlp_ratio=2.0).to(device)
        assert isinstance(block.ffn, MLP)
        assert block.ffn.layers[0].out_features == 32
        assert block.ffn.layers[-1].out_features == 16

    def test_gradient_flows_to_input(self, device):
        block = TransformerBlock(16, 4).to(device)
        x = torch.randn(2, 5, 16, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
