"""SLVT 模式的共享 generator 容器。"""

import torch
import torch.nn as nn

from mapping.base import Generator, MappingLayer, _prod


class Sequential(nn.Module):
    """SLVT 模式的共享 generator 容器。

    持有一个共享 generator，管理所有参数层的参数。
    weight 和 bias 沿两条独立的 flat 线分别切片。

    Args:
        *layers: 纯形状 MappingLayer（不能自带 generator），可混装非参数层
        generator_cls: Generator 子类
        **generator_kwargs: 透传给 generator 构造函数的参数
    """

    def __init__(self, *layers, generator_cls: type[Generator], **generator_kwargs):
        super().__init__()

        # 验证互斥：不能包含自带 generator 的层
        for i, layer in enumerate(layers):
            if isinstance(layer, MappingLayer) and hasattr(layer, 'generator'):
                raise ValueError(
                    f'Sequential 中的层不能自带 generator，但第 {i} 层 {layer} 已配置了 generator。'
                )

        self.layers = nn.ModuleList(layers)

        # 收集所有参数层的 weight/bias 大小，算切片边界。
        # bounds 列表只为参数层（MappingLayer）追加条目，非参数层跳过，
        # 这样 forward 中 param_idx 才能正确索引。
        w_total, b_total = 0, 0
        self.w_bounds = [0]
        self.b_bounds = [0]

        for layer in layers:
            if not isinstance(layer, MappingLayer):
                continue
            spec = layer.param_spec
            w_total += _prod(spec['weight'])
            self.w_bounds.append(w_total)
            if 'bias' in spec:
                b_total += _prod(spec['bias'])
            self.b_bounds.append(b_total)

        # 创建共享 generator
        full_spec = {'weight': (w_total,)}
        if b_total > 0:
            full_spec['bias'] = (b_total,)
        self.generator = generator_cls(full_spec, **generator_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_w, flat_b = self.generator()
        param_idx = 0

        for layer in self.layers:
            if isinstance(layer, MappingLayer):
                ws = self.w_bounds[param_idx]
                we = self.w_bounds[param_idx + 1]
                bs = self.b_bounds[param_idx]
                be = self.b_bounds[param_idx + 1]

                w_slice = flat_w[ws:we]
                b_slice = flat_b[bs:be] if flat_b is not None and be > bs else None

                x = layer.forward_with_params(x, w_slice, b_slice)
                param_idx += 1
            else:
                x = layer(x)

        return x
