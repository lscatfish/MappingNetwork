import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import tqdm
from ..mapping.mapping_net import MappingNetwork


class LWTTrainer:
    """
    Layer-wise Training (LWT / Ours†).

    Each layer/group of the target network gets its own latent vector z^(l)
    and MappingNetwork^(l). All layer losses are computed independently and aggregated.
    """

    def __init__(
        self,
        target_net,
        loss_fn,
        layer_latent_dims: dict,
        layer_alphas: dict = None,
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

        # Group target network parameters by layer name prefix
        self.param_groups = self._build_param_groups(target_net)

        # Create one MappingNetwork per layer/group using ModuleDict
        # so .to(device) propagates to all sub-modules
        self.layer_mappings = nn.ModuleDict()
        for group_name, group_size in self.param_groups:
            dim = layer_latent_dims.get(group_name, 64)
            alpha = layer_alphas.get(group_name, 0.01) if layer_alphas else 0.01
            self.layer_mappings[group_name] = MappingNetwork(
                group_size, dim, alpha=alpha,
            ).to(device)

        # Collect trainable params: all z's + loss lambda params
        trainable_params = [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        for mapping in self.layer_mappings.values():
            trainable_params.append(mapping.z)

        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)

    @staticmethod
    def _build_param_groups(target_net):
        """Group target net params by layer name prefix. Returns [(name, total_size), ...].

        Example: 'conv1.weight' and 'conv1.bias' both map to group 'conv1'.
        """
        groups = {}
        for name, param in target_net.named_parameters():
            base = name.split('.')[0]
            if base not in groups:
                groups[base] = 0
            groups[base] += param.numel()
        return list(groups.items())

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
            z_l = mapping.z
            start, end = offsets[group_name]

            # L_smooth^(l): ||nabla_z M^(l)(z^(l))||^2_F / P_l
            def mapping_fn(z_in):
                return torch.tanh(
                    (mapping.W_fixed + mapping.alpha * z_in.unsqueeze(0)) @ z_in
                    + mapping.b_fixed
                )

            jac = torch.func.jacfwd(mapping_fn)(z_l)
            l_smooth_total = l_smooth_total + torch.sum(jac ** 2) / jac.numel()

            # L_align^(l): 1 - cos(z^(l), mean(W_mod^(l)))
            W_mod = mapping.W_fixed + mapping.alpha * z_l.unsqueeze(0)
            W_m = W_mod.mean(dim=0)
            cos_sim = F.cosine_similarity(z_l.unsqueeze(0), W_m.unsqueeze(0))
            l_align_total = l_align_total + (1 - cos_sim.squeeze())

            # L_stab^(l): noise perturbation on z^(l)
            eps = torch.randn_like(z_l) * self.loss_fn.sigma_noise
            z_noisy_l = z_l + eps
            W_mod_n = mapping.W_fixed + mapping.alpha * z_noisy_l.unsqueeze(0)
            theta_noisy_l = torch.tanh(W_mod_n @ z_noisy_l + mapping.b_fixed)
            # Replace this layer's slice in the full theta
            theta_noisy = theta_hat.clone()
            theta_noisy[start:end] = theta_noisy_l
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
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%',
                })

        return total_loss / len(self.train_loader), 100. * correct / total

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

        return 100. * correct / total

    def save_checkpoint(self, results, epoch=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        suffix = f'_epoch{epoch}' if epoch else '_final'

        # Save dict of {layer_name: mapping.state_dict()} plus metadata
        checkpoint = {
            'target_net': self.checkpoint_metadata.get('target_net'),
            'training_strategy': self.checkpoint_metadata.get('training_strategy', 'lwt'),
            'layer_latent_dims': self.checkpoint_metadata.get('layer_latent_dims'),
            'layer_alphas': self.checkpoint_metadata.get('layer_alphas'),
            'alpha': self.checkpoint_metadata.get('alpha'),
            'sigma_noise': self.checkpoint_metadata.get('sigma_noise'),
            'state_dict': {
                name: mapping.state_dict()
                for name, mapping in self.layer_mappings.items()
            },
            'results': results,
            'epoch': epoch if epoch is not None else self.epochs,
        }
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        torch.save(checkpoint, path)

        results_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        return path

    def train(self):
        results = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate() if self.test_loader is not None else None
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': current_lr,
            })
            test_acc_str = f'{test_acc:.2f}%' if test_acc is not None else 'N/A'
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                  f'train_acc={train_acc:.2f}%, test_acc={test_acc_str}')

        path = self.save_checkpoint(results)
        print(f'Checkpoint saved to {path}')
        return results
