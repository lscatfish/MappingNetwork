"""LWT Trainer — Layer-wise Training (Ours†).

Each layer/group of the target network gets its own latent vector z^(l)
and ParameterGenerator^(l). All layer losses are computed independently and aggregated.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..factory import build_generator
from .base import BaseTrainer


class LWTTrainer(BaseTrainer):
    """Layer-wise Training (LWT / Ours†)。"""

    def __init__(
        self,
        target_net,
        loss_fn,
        layer_generators: dict,
        train_loader=None,
        test_loader=None,
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

        # Build layer order and compressed group sizes from the LRD-enabled target net
        self.layer_group_order = target_net.get_group_names()
        self.param_groups = [
            (name, target_net.get_group_param_size(name))
            for name in self.layer_group_order
        ]

        # Create one ParameterGenerator per layer/group using ModuleDict
        self.layer_generators = layer_generators
        self.layer_mappings: nn.ModuleDict[str, nn.Module] = nn.ModuleDict()
        for group_name, group_size in self.param_groups:
            if group_name not in layer_generators:
                raise ValueError(
                    f'Missing generator config for layer group: {group_name}'
                )
            config = dict(layer_generators[group_name])
            config['layer_name'] = group_name
            self.layer_mappings[group_name] = build_generator(
                config, target_total_params=group_size, device=device,
            )

        # 预缓存梯度裁剪参数，避免每个 batch 重复收集
        self._clip_params_cache = []
        for mapping in self.layer_mappings.values():
            self._clip_params_cache.extend(mapping.parameters())

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
        """所有 generator 参数 + loss lambda 参数。"""
        trainable = [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        for mapping in self.layer_mappings.values():
            trainable.extend(mapping.parameters())
        return trainable

    def _get_clip_params(self) -> list:
        return self._clip_params_cache

    def _generate_all_theta(self):
        """Concatenate all per-layer mapping outputs into full theta_hat."""
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

        L_stab 采样 n_stab_samples 次降低方差，与 SLVT 的 MappingLoss 行为一致。
        """
        l_stab_total = 0.0
        l_smooth_total = 0.0
        l_align_total = 0.0
        offsets = self._compute_offsets()
        n_stab_samples = self.loss_fn.n_stab_samples

        for group_name, mapping in self.layer_mappings.items():
            start, end = offsets[group_name]

            # L_smooth^(l) 与 L_align^(l) 由 generator 自行实现
            l_smooth_total = l_smooth_total + mapping.smooth_loss()
            l_align_total = l_align_total + mapping.align_loss()

            # L_stab^(l): 只扰动本层 z，替换到完整 theta 的对应切片。
            # theta_hat 必须 detach，避免未扰动层的梯度泄漏到其他层。
            # 多次采样降低方差，与 SLVT 的 MappingLoss.n_stab_samples 一致。
            l_stab_layer = 0.0
            for _ in range(n_stab_samples):
                theta_noisy = theta_hat.detach().clone()
                theta_noisy[start:end] = mapping.noisy_forward(self.loss_fn.sigma_noise)
                y_hat_noisy = self.target_net.functional_forward(x, theta_noisy)
                l_stab_layer = l_stab_layer + F.mse_loss(y_hat_noisy, y_hat.detach())
            l_stab_total = l_stab_total + l_stab_layer / n_stab_samples

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

        pbar = __import__('tqdm').tqdm(
            self.train_loader, desc=f'Epoch {epoch}/{self.epochs}'
        )
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
            self._clip_grads()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

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

    # ===== Checkpoint =====

    def _get_persistent_state(self) -> dict:
        return {
            name: mapping.persistent_state_dict()
            for name, mapping in self.layer_mappings.items()
        }

    def _load_persistent_state(self, state_dict: dict):
        for name, state in state_dict.items():
            self.layer_mappings[name].load_persistent_state_dict(state)

    def _get_loss_fn_state_dict(self):
        return self.loss_fn.state_dict()

    def _load_loss_fn_state_dict(self, state_dict):
        self.loss_fn.load_state_dict(state_dict)

    def _build_checkpoint(self, results, suffix, epoch, is_best) -> dict:
        ckpt = super()._build_checkpoint(results, suffix, epoch, is_best)
        ckpt['target_net'] = self.checkpoint_metadata.get('target_net')
        ckpt['training_strategy'] = self.checkpoint_metadata.get(
            'training_strategy', 'lwt'
        )
        ckpt['layer_generator_configs'] = copy.deepcopy(self.layer_generators)
        ckpt['layer_group_order'] = self.layer_group_order
        ckpt['lrd_config'] = self.checkpoint_metadata.get('lrd_config')
        ckpt['alpha'] = self.checkpoint_metadata.get('alpha')
        ckpt['sigma_noise'] = self.checkpoint_metadata.get('sigma_noise')
        return ckpt
