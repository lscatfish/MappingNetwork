"""Optimizer and scheduler builders used by trainers."""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def build_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    """Build a PyTorch optimizer by name."""
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
    """Build a PyTorch LR scheduler by name."""
    name = name.lower()
    if name in ('cosine_annealing', 'cosine'):
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == 'step':
        step_size = kwargs.get('step_size', max(1, epochs // 3))
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f'Unsupported scheduler: {name}')
