"""generator 积木基类：Block。"""

import torch.nn as nn


class _BlockMeta(type):
    """Block 元类：__init__ 结束后自动执行 init_weights() 并冻结全部参数。"""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.init_weights()
        instance._freeze()
        return instance


class Block(nn.Module, metaclass=_BlockMeta):
    """可组合的 generator 积木基类。

    继承本类后写法与 torch.nn.Module 完全一致：在 __init__ 中创建
    子模块，在 forward 中组合调用（支持残差等任意结构）。构造结束
    后框架自动执行 init_weights() 并递归冻结全部参数
    (requires_grad_(False))，用户无需手动处理。

    子类可重载 init_weights() 自定义初始化；默认为 no-op
    （组合块不重新初始化已就位的子块）。
    """

    def init_weights(self) -> None:
        """初始化钩子，默认 no-op。子类可重载。"""

    def _freeze(self) -> None:
        """递归冻结全部参数（幂等）。"""
        for p in self.parameters():
            p.requires_grad_(False)
