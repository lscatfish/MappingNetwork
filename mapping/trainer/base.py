"""BaseTrainer：通用训练循环、评估、checkpoint、日志。"""

import json
import logging
import os

import torch
import tqdm

from mapping.loss import MappingLoss

from .optim_utils import build_optimizer, build_scheduler


class BaseTrainer:
    """训练器基类。

    Args:
        net: 主干网络（nn.Module）
        loss_fn: MappingLoss 实例
        generators: 需要优化的 Generator 列表
        train_loader: 训练数据 DataLoader
        test_loader: 测试数据 DataLoader（可选）
        lr: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        min_lr: 最小学习率
        device: 设备
        log_interval: 日志打印间隔（batch 数）
        checkpoint_dir: checkpoint 保存目录
        experiment_name: 实验名称
        save_interval: 中间 checkpoint 保存间隔（epoch）
        optimizer_name: 优化器名称
        scheduler_name: 调度器名称
        grad_clip_norm: 梯度裁剪范数（None 表示不裁剪）
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss_fn: MappingLoss,
        generators: list,
        train_loader,
        test_loader=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'train',
        save_interval: int = 1,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine_annealing',
        grad_clip_norm: float | None = 1.0,
    ):
        self.net = net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.generators = generators
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.save_interval = save_interval
        self.grad_clip_norm = grad_clip_norm
        self.best_test_acc = -1.0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = self._setup_logger()

        trainable_params = self._collect_trainable_params()
        self.optimizer = build_optimizer(
            trainable_params, optimizer_name, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = build_scheduler(
            self.optimizer, scheduler_name, epochs=epochs, min_lr=min_lr,
        )

    def _collect_trainable_params(self) -> list:
        params = []
        for gen in self.generators:
            params.extend(gen.parameters())
        params.extend([
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ])
        return params

    def _setup_logger(self):
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}.log')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播得到 logits。子类可重写。"""
        return self.net(x)

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            logits = self._get_logits(x)
            loss, _ = self.loss_fn(logits, y, self.generators)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                params = [p for p in self._collect_trainable_params() if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%',
                })

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self) -> float | None:
        if self.test_loader is None:
            return None
        self.net.eval()
        correct = 0
        total = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self._get_logits(x)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        return 100.0 * correct / total

    def save_checkpoint(self, results, suffix='_final', epoch=None, is_best=False):
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        checkpoint = {
            'net_state_dict': self.net.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_acc': self.best_test_acc,
            'results': results,
            'epoch': epoch if epoch is not None else self.epochs,
            'is_best': is_best,
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_test_acc = checkpoint.get('best_test_acc', -1.0)
        return checkpoint.get('epoch', 0), checkpoint.get('results', [])

    def save_results(self, results):
        results_path = os.path.join(
            self.checkpoint_dir, f'{self.experiment_name}_results.json'
        )
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results_path

    def train(self, start_epoch: int = 1) -> list[dict]:
        self.logger.info(
            f'Start training: {self.experiment_name}, '
            f'device={self.device}, epochs={self.epochs}'
        )
        results = []
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate()
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': current_lr,
            }
            results.append(epoch_result)

            test_str = f'{test_acc:.2f}%' if test_acc is not None else 'N/A'
            self.logger.info(
                f'Epoch {epoch}: loss={train_loss:.4f}, '
                f'train_acc={train_acc:.2f}%, test_acc={test_str}, lr={current_lr:.6f}'
            )

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self.save_checkpoint(results, suffix=f'_epoch{epoch}', epoch=epoch)

            if test_acc is not None and test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.save_checkpoint(results, suffix='_best', epoch=epoch, is_best=True)

        self.save_checkpoint(results, suffix='_final', epoch=self.epochs)
        self.save_results(results)
        return results
