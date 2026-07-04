"""
Unified training entry point for Mapping Networks.

Usage:
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.slvt import SLVTTrainer
from mapping_network.trainer.lwt import LWTTrainer

TARGET_NET_MAP = {
    'cnn2': CNN2,
    'cnn1': CNN1,
    'cnn1_3conv': CNN1_3Conv,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.seed:
        cfg['seed'] = args.seed

    if 'seed' in cfg:
        set_seed(cfg['seed'])

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'])

    # Target network
    target_cls = TARGET_NET_MAP[cfg['target_net']]
    target_net = target_cls()
    print(f'Target network: {cfg["target_net"]}, '
          f'params: {target_net.get_total_params():,}')

    # Loss
    loss_fn = MappingLoss(sigma_noise=cfg.get('sigma_noise', 0.01)).to(device)

    exp_name = f"{cfg['target_net']}_{cfg['training_strategy']}"

    if cfg['training_strategy'] == 'slvt':
        mapping = MappingNetwork(
            target_net.get_total_params(),
            cfg['latent_dim'],
            alpha=cfg.get('alpha', 0.01),
        ).to(device)
        print(f'Latent dim: {cfg["latent_dim"]}')
        print(f'Trainable: {sum(p.numel() for p in mapping.parameters() if p.requires_grad):,}')
        print(f'Fixed mapping weights: {mapping.W_fixed.numel():,}')

        trainer = SLVTTrainer(
            mapping, target_net, loss_fn,
            train_loader, test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=cfg.get('checkpoint_dir', 'checkpoints'),
            experiment_name=exp_name,
        )
    elif cfg['training_strategy'] == 'lwt':
        trainer = LWTTrainer(
            target_net, loss_fn,
            cfg['layer_latent_dims'],
            layer_alphas=cfg.get('layer_alphas'),
            train_loader=train_loader,
            test_loader=test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=cfg.get('checkpoint_dir', 'checkpoints'),
            experiment_name=exp_name,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg['training_strategy']}")

    results = trainer.train()
    final_acc = results[-1]['test_acc']
    print(f'\nFinal test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
