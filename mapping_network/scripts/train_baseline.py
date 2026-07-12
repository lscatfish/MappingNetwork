"""Baseline trainer and CLI — trains a target network directly (without Mapping Network).

Usage:
  # 使用配置文件
  uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml

  # 使用命令行参数
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 1 --device cpu
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from mapping_network.factory import TARGET_NET_MAP
from mapping_network.trainer.base import BaseTrainer
from mapping_network.data import get_mnist_loaders


class BaselineTrainer(BaseTrainer):
    """直接训练目标网络（不使用 Mapping Network）。"""

    def __init__(
        self,
        target_net_name: str,
        train_loader,
        test_loader=None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'baseline',
        checkpoint_metadata: dict = None,
        save_interval: int = 1,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine_annealing',
        append_log: bool = False,
    ):
        self.target_net_name = target_net_name
        self.model = TARGET_NET_MAP[target_net_name]().to(device)

        super().__init__(
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            min_lr=min_lr,
            device=device,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            checkpoint_metadata=checkpoint_metadata,
            save_interval=save_interval,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            append_log=append_log,
        )

    def _get_trainable_params(self) -> list:
        return list(self.model.parameters())

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = __import__('tqdm').tqdm(
            self.train_loader, desc=f'Epoch {epoch}/{self.epochs}'
        )
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = y_hat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0 * correct / total:.2f}%'})

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self):
        if self.test_loader is None:
            return None
        self.model.eval()
        correct = 0
        total = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            _, pred = y_hat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

    # ===== Checkpoint =====

    def _get_persistent_state(self) -> dict:
        return self.model.state_dict()

    def _load_persistent_state(self, state_dict: dict):
        self.model.load_state_dict(state_dict)

    def _build_checkpoint(self, results, suffix, epoch, is_best) -> dict:
        ckpt = super()._build_checkpoint(results, suffix, epoch, is_best)
        ckpt['type'] = 'baseline'
        ckpt['target_net'] = self.target_net_name
        ckpt['epochs'] = self.epochs
        ckpt['final_test_acc'] = results[-1].get('test_acc') if results else None
        return ckpt


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--target', type=str, default=None, choices=['cnn1', 'cnn2', 'cnn1_3conv'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}

    target = args.target if args.target is not None else cfg.get('target')
    epochs = args.epochs if args.epochs is not None else cfg.get('epochs', 30)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 64)
    lr = args.lr if args.lr is not None else cfg.get('lr', 0.001)
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    device = args.device if args.device is not None else cfg.get('device', 'cuda')
    checkpoint_dir = (
        args.checkpoint_dir if args.checkpoint_dir is not None
        else cfg.get('checkpoint_dir', 'checkpoints')
    )
    save_interval = (
        args.save_interval if args.save_interval is not None
        else cfg.get('save_interval', 1)
    )

    if target is None:
        parser.error('--target is required when no config file is provided')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_name = f'{target}_baseline'
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)

    train_loader, test_loader = get_mnist_loaders(batch_size, root='./data')

    trainer = BaselineTrainer(
        target_net_name=target,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=lr,
        weight_decay=0.0001,
        epochs=epochs,
        device=device,
        log_interval=100,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        checkpoint_metadata={
            'target_net': target,
        },
        save_interval=save_interval,
        append_log=args.resume is not None,
    )

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainer.logger.info(f'Training {target} baseline: {total_params:,} params')

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        trainer.logger.info(f'Resumed from {args.resume}, starting at epoch {start_epoch}')
    else:
        start_epoch = 1

    results = trainer.train(start_epoch=start_epoch)
    final_acc = results[-1]['test_acc']
    print(f'\nFinal test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
