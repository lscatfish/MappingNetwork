"""低秩分解 (LRD) 辅助模块。"""

import torch
import torch.nn as nn


class LRDLayer(nn.Module):
    """低秩分解辅助模块。

    供用户在 generator.forward() 中调用，将 flat 输出分解为 U, V
    并重构为完整 weight 矩阵 W = U @ V^T。

    Args:
        m: 原始 weight 的行数
        n: 原始 weight 的列数
        rank: 低秩分解的秩 r
    """

    def __init__(self, m: int, n: int, rank: int):
        super().__init__()
        self.m = m
        self.n = n
        self.rank = rank

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        """将 flat 张量分解为 U, V 并重构为 W = U @ V^T。

        Args:
            flat: 形状为 (m * rank + n * rank,) 的一维张量

        Returns:
            weight: 形状为 (m, n) 的完整权重矩阵
        """
        m, n, r = self.m, self.n, self.rank
        U = flat[:m * r].reshape(m, r)
        V = flat[m * r:].reshape(n, r)
        return U @ V.T

    def extra_repr(self) -> str:
        return f'm={self.m}, n={self.n}, rank={self.rank}'
