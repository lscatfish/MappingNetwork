import copy
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from ..factory import build_generator
from ..generators.base import ParameterGenerator
from .optim_utils import build_optimizer, build_scheduler


class LWTTrainer:
    """
    Layer-wise Training (LWT / Ours†).

    Each layer/group of the target network gets its own latent vector z^(l)
    and ParameterGenerator^(l). All layer losses are computed independently and aggregated.
    """

    def __init__(
        self,
        target_net,
        loss_fn,
        layer_generators: dict,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'lwt',
        checkpoint_metadata: dict = None,
        save_interval: int = 1,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine_annealing',
        append_log: bool = False,
    ):
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

        # Build layer order and compressed group sizes from the LRD-enabled target net
        self.layer_group_order = target_net.get_group_names()
        self.param_groups = [
            (name, target_net.get_group_param_size(name)) for name in self.layer_group_order
        ]

        # Create one ParameterGenerator per layer/group using ModuleDict
        # so .to(device) propagates to all sub-modules
        self.layer_generators = layer_generators
        self.layer_mappings: nn.ModuleDict[str, ParameterGenerator] = nn.ModuleDict()
        for group_name, group_size in self.param_groups:
            if group_name not in layer_generators:
                raise ValueError(f'Missing generator config for layer group: {group_name}')
            config = layer_generators[group_name]
            self.layer_mappings[group_name] = build_generator(
                config['type'],
                {
                    'target_total_params': group_size,
                    'latent_dim': config['latent_dim'],
                    'alpha': config['alpha'],
                },
                device=device,
            )

        # Collect trainable params: all generator params + loss lambda params
        trainable_params = [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        for mapping in self.layer_mappings.values():
            trainable_params.extend(mapping.parameters())

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
        logger.handlers = []

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}.log')
        log_mode = 'a' if self.append_log else 'w'
        file_handler = logging.FileHandler(log_path, mode=log_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _generate_all_theta(self):
        """Concatenate all per-layer mapping outputs into full theta_hat.

        Each layer's MappingNetwork produces theta^(l) of size P_l.
        Concatenating in group order yields the full parameter vector theta_hat.
        """
        all_theta = []
        for group_name, _ in self.param_groups:
            all_theta.append(self.layer_mappings[group_name]())
        return torch.cat(all_theta)

    def _compute_offsets(self):
        """Compute (start, end) indices for each group in the concatenated theta_hat."""
        offsets = {}
        offset = 0
        for group_name, group_size in self.param_groups:
            offsets[group_name] = (offset, offset + group_size)
            offset += group_size
        return offsets

    def _compute_layerwise_reg_loss(self, x, y, theta_hat, y_hat):
        """Compute per-layer L_smooth + L_align + L_stab, then sum.

        Each term is computed independently per layer and aggregated.
        Uses functional_forward for all forward passes (no .data.copy_()).
        """
        l_stab_total = 0.0
        l_smooth_total = 0.0
        l_align_total = 0.0
        offsets = self._compute_offsets()

        for group_name, mapping in self.layer_mappings.items():
            start, end = offsets[group_name]

            # L_smooth^(l) 与 L_align^(l) 由 generator 自行实现
            l_smooth_total = l_smooth_total + mapping.smooth_loss()
            l_align_total = l_align_total + mapping.align_loss()

            # L_stab^(l): 只扰动本层 z，替换到完整 theta 的对应切片。
            # theta_hat 必须 detach，避免未扰动层的梯度泄漏到其他层。
            theta_noisy = theta_hat.detach().clone()
            theta_noisy[start:end] = mapping.noisy_forward(self.loss_fn.sigma_noise)
            y_hat_noisy = self.target_net.functional_forward(x, theta_noisy)
            l_stab_total = l_stab_total + F.mse_loss(y_hat_noisy, y_hat.detach())

        # Weighted sum with sigmoid-gated lambdas
        l_st = torch.sigmoid(self.loss_fn.lambda_st)
        l_sm = torch.sigmoid(self.loss_fn.lambda_sm)
        l_al = torch.sigmoid(self.loss_fn.lambda_al)

        return l_st * l_stab_total + l_sm * l_smooth_total + l_al * l_align_total

    def train_epoch(self, epoch):
        self.target_net.train()
        for mapping in self.layer_mappings.values():
            mapping.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # Generate full theta_hat
            theta_hat = self._generate_all_theta()

            # Task loss via functional forward
            y_hat = self.target_net.functional_forward(x, theta_hat)
            l_task = F.cross_entropy(y_hat, y)

            # Per-layer regularization losses
            reg_loss = self._compute_layerwise_reg_loss(x, y, theta_hat, y_hat)

            loss = l_task + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

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
        self.target_net.eval()
        for mapping in self.layer_mappings.values():
            mapping.eval()

        theta_hat = self._generate_all_theta()
        correct = 0
        total = 0

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.target_net.functional_forward(x, theta_hat)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return 100.0 * correct / total

    def save_checkpoint(self, results, suffix='_final', epoch=None, is_best=False):
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')

        # Save dict of {layer_name: mapping.state_dict()} plus metadata
        checkpoint = {
            'target_net': self.checkpoint_metadata.get('target_net'),
            'training_strategy': self.checkpoint_metadata.get('training_strategy', 'lwt'),
            'layer_generator_configs': copy.deepcopy(self.layer_generators),
            'layer_group_order': self.layer_group_order,
            'lrd_config': self.checkpoint_metadata.get('lrd_config'),
            'alpha': self.checkpoint_metadata.get('alpha'),
            'sigma_noise': self.checkpoint_metadata.get('sigma_noise'),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'state_dict': {
                name: mapping.state_dict() for name, mapping in self.layer_mappings.items()
            },
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
        for name, state in checkpoint['state_dict'].items():
            self.layer_mappings[name].load_state_dict(state)
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
            f'Start LWT training: {self.experiment_name}, '
            f'device={self.device}, epochs={self.epochs}'
        )
        results = list(getattr(self, 'results', []))
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate() if self.test_loader is not None else None
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            results.append(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'lr': current_lr,
                }
            )
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

        final_path = self.save_checkpoint(results, suffix='_final', epoch=self.epochs)
        results_path = self.save_results(results)
        self.logger.info(f'Final checkpoint saved to {final_path}')
        self.logger.info(f'Results JSON saved to {results_path}')
        return results
