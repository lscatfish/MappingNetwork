"""预置积木块：TransformerBlock（pre-norm）。"""

import torch
import torch.nn as nn

from mapping.generator.block import Block
from mapping.generator.mlp import MLP


class TransformerBlock(Block):
    """pre-norm Transformer 块。

    结构：x + attn(norm1(x))，再 x + ffn(norm2(x))。
    输入形状 (B, L, D)，输出形状不变。内部全部参数固定。

    Args:
        dim: 特征维度 D
        num_heads: 注意力头数（须整除 dim）
        mlp_ratio: FFN 隐藏层倍数（默认 4.0）
        dropout: 注意力 dropout（默认 0.0）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP([dim, int(dim * mlp_ratio), dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))
