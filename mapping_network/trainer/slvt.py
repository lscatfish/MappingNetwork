import json
import logging
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from mapping_network.generators.base import ParameterGenerator

from .optim_utils import build_optimizer, build_scheduler


class SLVTTrainer:
    """
    Single Latent Vector Training (SLVT / Ours*).

    一个 latent vector z 生成全部目标网络参数。
    使用函数式前向保持梯度完整。
    """

    def __init__(
        self,
        mapping_net: ParameterGenerator,
        target_net,
        loss_fn,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'slvt',
        checkpoint_metadata: dict = None,
        save_interval: int = 1,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine_annealing',
        append_log: bool = False,
    ):
        self.mapping_net = mapping_net.to(device)
        self.target_net = target_net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.save_interval = save_interval
        self.append_log = append_log
        self.best_test_acc = -1.0

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = self._setup_logger()

        # 只更新生成网络的可训练参数和 λ (MappingNet 权重固定)
        trainable_params = list(self.mapping_net.parameters()) + [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        self.optimizer = build_optimizer(
            trainable_params, optimizer_name, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = build_scheduler(
            self.optimizer, scheduler_name, epochs=epochs, min_lr=min_lr
        )

    def _setup_logger(self):
        """设置日志同时输出到控制台和文件。"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        # 避免重复添加 handler（如多次实例化）
        logger.handlers = []

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件输出
        log_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}.log')
        log_mode = 'a' if self.append_log else 'w'
        file_handler = logging.FileHandler(log_path, mode=log_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def train_epoch(self, epoch):
        self.mapping_net.train()
        self.target_net.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # 1. 从 z 生成参数
            theta_hat = self.mapping_net()

            # 2. 计算损失 (函数式前向)
            loss, losses_dict = self.loss_fn(
                theta_hat,
                self.mapping_net,
                self.target_net,
                x,
                y,
            )

            # 3. 计算准确率（在 optimizer.step() 之前，使用当前 theta_hat）
            with torch.no_grad():
                y_hat = self.target_net.functional_forward(x, theta_hat)
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(
                    {
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.0 * correct / total:.2f}%',
                    }
                )

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self):
        if self.test_loader is None:
            return None
        self.mapping_net.eval()
        self.target_net.eval()
        correct = 0
        total = 0

        theta_hat = self.mapping_net()
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.target_net.functional_forward(x, theta_hat)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return 100.0 * correct / total

    def save_checkpoint(self, results, suffix='_final', epoch=None, is_best=False):
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')

        checkpoint = {
            'target_net': self.checkpoint_metadata.get('target_net'),
            'training_strategy': self.checkpoint_metadata.get('training_strategy', 'slvt'),
            'generator_type': self.checkpoint_metadata.get('generator_type', 'linear'),
            'latent_dim': self.checkpoint_metadata.get('latent_dim'),
            'alpha': self.checkpoint_metadata.get('alpha'),
            'sigma_noise': self.checkpoint_metadata.get('sigma_noise'),
            'lrd_config': self.checkpoint_metadata.get('lrd_config'),
            'generator_config': self.checkpoint_metadata.get('generator_config'),
            'generator_state_dict': self.mapping_net.persistent_state_dict(),
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
        self.mapping_net.load_persistent_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_test_acc = checkpoint.get('best_test_acc', -1.0)
        self.results = checkpoint.get('results', [])
        if 'loss_fn_state_dict' in checkpoint:
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        return checkpoint.get('epoch', 0)

    def save_results(self, results):
        """保存训练结果到 JSON。"""
        results_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results_path

    def train(self, start_epoch=1):
        self.logger.info(
            f'Start SLVT training: {self.experiment_name}, '
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
                f'train_acc={train_acc:.2f}%, test_acc={test_acc_str}, lr={current_lr:.6f}'
            )
            self.logger.info(msg)

            # 保存中间模型
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                inter_path = self.save_checkpoint(results, suffix=f'_epoch{epoch}', epoch=epoch)
                self.logger.info(f'Intermediate checkpoint saved to {inter_path}')

            # 保存最优模型
            if test_acc is not None and test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                best_path = self.save_checkpoint(results, suffix='_best', epoch=epoch, is_best=True)
                self.logger.info(f'New best test_acc={test_acc:.2f}%, saved to {best_path}')

        # 保存最终 checkpoint
        final_path = self.save_checkpoint(results, suffix='_final', epoch=self.epochs)
        results_path = self.save_results(results)
        self.logger.info(f'Final checkpoint saved to {final_path}')
        self.logger.info(f'Results JSON saved to {results_path}')
        return results
