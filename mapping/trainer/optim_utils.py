"""Optimizer and scheduler builders."""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, StepLR


def build_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    name = name.lower()
    if name in ('adamw', 'adam_w'):
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f'Unsupported optimizer: {name}')


def build_scheduler(optimizer, name: str, epochs: int, min_lr: float, **kwargs):
    name = name.lower()
    if name in ('cosine_annealing', 'cosine'):
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == 'step':
        step_size = kwargs.get('step_size', max(1, epochs // 3))
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name in ('warmup_cosine', 'cosine_warmup'):
        warmup_epochs = kwargs.get('warmup_epochs', max(1, epochs // 10))

        def warmup_fn(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup = LambdaLR(optimizer, warmup_fn)
        cosine = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
        )
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    raise ValueError(f'Unsupported scheduler: {name}')
