"""SLVTTrainer：Sequential 共享 generator 网络的训练。"""

from mapping.sequential import Sequential

from .base import BaseTrainer


class SLVTTrainer(BaseTrainer):
    """SLVT 训练器。

    net 为 Sequential 实例，generators 自动从 net.generator 收集。
    """

    def __init__(self, net: Sequential, loss_fn, train_loader, **kwargs):
        generators = [net.generator]
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            generators=generators,
            train_loader=train_loader,
            **kwargs,
        )
