"""mapping.trainer — 训练器模块。"""

from mapping.trainer.base import BaseTrainer
from mapping.trainer.lwt import LWTTrainer, collect_generators
from mapping.trainer.slvt import SLVTTrainer

__all__ = [
    'BaseTrainer',
    'SLVTTrainer',
    'LWTTrainer',
    'collect_generators',
]
