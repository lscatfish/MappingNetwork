"""LWTTrainer：逐层 generator 网络的训练。"""

import torch.nn as nn

from mapping.base import MappingLayer

from .base import BaseTrainer


def collect_generators(net: nn.Module) -> list:
    """从网络中收集所有 MappingLayer 的 generator。"""
    generators = []
    for module in net.modules():
        if isinstance(module, MappingLayer) and hasattr(module, 'generator'):
            generators.append(module.generator)
    return generators


class LWTTrainer(BaseTrainer):
    """LWT 训练器。

    net 为用户自定义 Module（内含多个带 generator 的 MappingLayer），
    generators 自动从 net 中收集。
    """

    def __init__(self, net: nn.Module, loss_fn, train_loader, **kwargs):
        generators = collect_generators(net)
        if not generators:
            raise ValueError('网络中未找到带 generator 的 MappingLayer')
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            generators=generators,
            train_loader=train_loader,
            **kwargs,
        )
