"""Trainer 基类，封装公共的训练循环、日志、checkpoint 逻辑。

SLVTTrainer / LWTTrainer / BaselineTrainer 继承此类，
只需实现 train_epoch / evaluate / _get_persistent_state / _load_persistent_state。
"""

import json
import logging
import os

import torch
import tqdm

from .optim_utils import build_optimizer, build_scheduler


class BaseTrainer:
    """训练器基类。

    子类必须实现：
        - train_epoch(epoch) -> (train_loss, train_acc)
        - evaluate() -> test_acc | None
        - _get_persistent_state() -> dict  （用于 save_checkpoint）
        - _load_persistent_state(state_dict)  （用于 load_checkpoint）

    子类可选重写：
        - _get_clip_params() -> list  （返回需要梯度裁剪的参数）
    """

    def __init__(
        self,
        train_loader,
        test_loader=None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'train',
        checkpoint_metadata: dict = None,
        save_interval: int = 1,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine_annealing',
        append_log: bool = False,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.save_interval = save_interval
        self.append_log = append_log
        self.best_test_acc = -1.0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = self._setup_logger()

        # 子类在调用 super().__init__ 前应设置好 trainable_params
        trainable_params = self._get_trainable_params()
        self.optimizer = build_optimizer(
            trainable_params, optimizer_name, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = build_scheduler(
            self.optimizer, scheduler_name, epochs=epochs, min_lr=min_lr,
            warmup_epochs=self.checkpoint_metadata.get('warmup_epochs', max(1, epochs // 10)),
        )

    # ===== 子类必须实现 =====

    def _get_trainable_params(self) -> list:
        """返回需要优化的参数列表。子类在 super().__init__ 前确保相关模块已创建。"""
        raise NotImplementedError

    def train_epoch(self, epoch: int):
        """训练一个 epoch，返回 (train_loss, train_acc)。"""
        raise NotImplementedError

    def evaluate(self):
        """在测试集上评估，返回 test_acc 或 None。"""
        raise NotImplementedError

    def _get_persistent_state(self) -> dict:
        """返回需要持久化到 checkpoint 的 state_dict。"""
        raise NotImplementedError

    def _load_persistent_state(self, state_dict: dict):
        """从 checkpoint 的 state_dict 恢复模型参数。"""
        raise NotImplementedError

    # ===== 子类可选重写 =====

    def _get_clip_params(self) -> list:
        """返回需要梯度裁剪的参数。默认返回空列表（不裁剪）。"""
        return []

    def _build_checkpoint(self, results, suffix, epoch, is_best) -> dict:
        """构建 checkpoint dict。子类可重写以添加额外字段。"""
        return {
            'loss_fn_state_dict': self._get_loss_fn_state_dict(),
            'state_dict': self._get_persistent_state(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_acc': self.best_test_acc,
            'results': results,
            'epoch': epoch if epoch is not None else self.epochs,
            'is_best': is_best,
        }

    def _load_checkpoint_extra(self, checkpoint: dict):
        """从 checkpoint 恢复额外状态（如 loss_fn）。子类可重写。"""
        pass

    def _get_loss_fn_state_dict(self):
        """返回 loss_fn 的 state_dict，用于 checkpoint。子类可重写。"""
        return None

    # ===== 公共逻辑 =====

    def _setup_logger(self):
        """设置日志同时输出到控制台和文件。"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers = []

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}.log')
        log_mode = 'a' if self.append_log else 'w'
        file_handler = logging.FileHandler(log_path, mode=log_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _clip_grads(self):
        """梯度裁剪，使用子类 _get_clip_params 返回的参数。"""
        clip_params = self._get_clip_params()
        if clip_params:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

    def save_checkpoint(self, results, suffix='_final', epoch=None, is_best=False):
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        checkpoint = self._build_checkpoint(results, suffix, epoch, is_best)
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._load_persistent_state(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_test_acc = checkpoint.get('best_test_acc', -1.0)
        self.results = checkpoint.get('results', [])
        if checkpoint.get('loss_fn_state_dict') is not None:
            self._load_loss_fn_state_dict(checkpoint['loss_fn_state_dict'])
        self._load_checkpoint_extra(checkpoint)
        return checkpoint.get('epoch', 0)

    def _load_loss_fn_state_dict(self, state_dict):
        """从 state_dict 恢复 loss_fn。子类可重写。"""
        pass

    def save_results(self, results):
        """保存训练结果到 JSON。"""
        results_path = os.path.join(
            self.checkpoint_dir, f'{self.experiment_name}_results.json'
        )
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results_path

    def train(self, start_epoch=1):
        self.logger.info(
            f'Start training: {self.experiment_name}, '
            f'device={self.device}, epochs={self.epochs}'
        )
        results = list(getattr(self, 'results', []))
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate() if self.test_loader is not None else None
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
            test_acc_str = f'{test_acc:.2f}%' if test_acc is not None else 'N/A'
            msg = (
                f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                f'train_acc={train_acc:.2f}%, test_acc={test_acc_str}, '
                f'lr={current_lr:.6f}'
            )
            self.logger.info(msg)

            # 保存中间模型
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                inter_path = self.save_checkpoint(
                    results, suffix=f'_epoch{epoch}', epoch=epoch
                )
                self.logger.info(f'Intermediate checkpoint saved to {inter_path}')

            # 保存最优模型
            if test_acc is not None and test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                best_path = self.save_checkpoint(
                    results, suffix='_best', epoch=epoch, is_best=True
                )
                self.logger.info(
                    f'New best test_acc={test_acc:.2f}%, saved to {best_path}'
                )

        # 保存最终 checkpoint
        final_path = self.save_checkpoint(results, suffix='_final', epoch=self.epochs)
        results_path = self.save_results(results)
        self.logger.info(f'Final checkpoint saved to {final_path}')
        self.logger.info(f'Results JSON saved to {results_path}')
        return results
