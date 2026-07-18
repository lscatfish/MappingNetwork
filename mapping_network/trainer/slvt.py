"""SLVT Trainer — Single Latent Vector Training (Ours*).

一个 latent vector z 生成全部目标网络参数。
使用函数式前向保持梯度完整。
"""

import torch

from ..generators.base import ParameterGenerator
from .base import BaseTrainer


class SLVTTrainer(BaseTrainer):
    """Single Latent Vector Training (SLVT / Ours*)。"""

    def __init__(
        self,
        mapping_net: ParameterGenerator,
        target_net,
        loss_fn,
        train_loader,
        test_loader=None,
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
        """生成网络的可训练参数 + loss lambda 参数。"""
        return list(self.mapping_net.parameters()) + [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]

    def _get_clip_params(self) -> list:
        return list(self.mapping_net.parameters())

    def train_epoch(self, epoch):
        self.mapping_net.train()
        self.target_net.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = __import__('tqdm').tqdm(
            self.train_loader, desc=f'Epoch {epoch}/{self.epochs}'
        )
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # 1. 从 z 生成参数
            theta_hat = self.mapping_net()

            # 2. 计算损失 (函数式前向)
            loss, losses_dict = self.loss_fn(
                theta_hat, self.mapping_net, self.target_net, x, y,
            )

            # 3. 计算准确率
            with torch.no_grad():
                y_hat = self.target_net.functional_forward(x, theta_hat)
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            # 4. 反向传播 + 梯度裁剪
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_grads()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%',
                })

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

    # ===== Checkpoint =====

    def _get_persistent_state(self) -> dict:
        return self.mapping_net.persistent_state_dict()

    def _load_persistent_state(self, state_dict: dict):
        self.mapping_net.load_persistent_state_dict(state_dict)

    def _get_loss_fn_state_dict(self):
        return self.loss_fn.state_dict()

    def _load_loss_fn_state_dict(self, state_dict):
        self.loss_fn.load_state_dict(state_dict)

    def _build_checkpoint(self, results, suffix, epoch, is_best) -> dict:
        ckpt = super()._build_checkpoint(results, suffix, epoch, is_best)
        ckpt['target_net'] = self.checkpoint_metadata.get('target_net')
        ckpt['training_strategy'] = self.checkpoint_metadata.get(
            'training_strategy', 'slvt'
        )
        ckpt['gen_config'] = self.checkpoint_metadata.get('gen_config')
        ckpt['latent_dim'] = self.checkpoint_metadata.get('latent_dim')
        ckpt['alpha'] = self.checkpoint_metadata.get('alpha')
        ckpt['sigma_noise'] = self.checkpoint_metadata.get('sigma_noise')
        ckpt['lrd_config'] = self.checkpoint_metadata.get('lrd_config')
        return ckpt
