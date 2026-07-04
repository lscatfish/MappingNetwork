import os
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import tqdm


class SLVTTrainer:
    """
    Single Latent Vector Training (SLVT / Ours*).

    一个 latent vector z 生成全部目标网络参数。
    使用函数式前向保持梯度完整。
    """

    def __init__(
        self,
        mapping_net,
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

        # 只更新 z 和 λ (MappingNet 权重固定)
        trainable_params = [
            self.mapping_net.z,
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)

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

            # 2. 计算噪声版本用于 L_stab
            eps = torch.randn_like(self.mapping_net.z) * self.loss_fn.sigma_noise
            z_noisy = self.mapping_net.z + eps
            W_mod_noisy = self.mapping_net.W_fixed + self.mapping_net.alpha * z_noisy.unsqueeze(0)
            theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + self.mapping_net.b_fixed)

            # 3. 计算损失 (函数式前向)
            loss, losses_dict = self.loss_fn(
                self.mapping_net.z, theta_hat, theta_noisy,
                self.mapping_net, self.target_net, x, y,
            )

            # 4. 计算准确率（在 optimizer.step() 之前，使用当前 theta_hat）
            with torch.no_grad():
                y_hat = self.target_net.functional_forward(x, theta_hat)
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            # 5. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                })

        return total_loss / len(self.train_loader), 100. * correct / total

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

        return 100. * correct / total

    def save_checkpoint(self, results, epoch=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        suffix = f"_epoch{epoch}" if epoch else "_final"
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        torch.save(self.mapping_net.state_dict(), path)

        # 同时保存结果
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
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': current_lr,
            }
            results.append(epoch_result)
            test_acc_str = f'{test_acc:.2f}%' if test_acc is not None else 'N/A'
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                  f'train_acc={train_acc:.2f}%, test_acc={test_acc_str}')

        # 保存最终 checkpoint
        path = self.save_checkpoint(results)
        print(f'Checkpoint saved to {path}')
        return results
