"""
Unified training entry point for Mapping Networks.

Usage:
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapping_network.factory import build_generator, build_target_net
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_experiment_name(cfg):
    target = cfg['target_net']
    strategy = cfg['training_strategy']
    return f'{target}_{strategy}'


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
    print(f'Using device: {device}')
    print(
        f'Strategy: {cfg["training_strategy"]}, Target: {cfg["target_net"]}, Epochs: {cfg["epochs"]}'
    )

    # Data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'])

    lrd_config = cfg.get('lrd', {})

    # Merge per-layer lrd_rank overrides into global LRDConfig
    if cfg['training_strategy'] == 'lwt':
        layer_ranks = {}
        for name, gen_cfg in cfg['layer_generators'].items():
            if 'lrd_rank' in gen_cfg:
                layer_ranks[name] = gen_cfg['lrd_rank']
        if layer_ranks:
            lrd_config = {
                **lrd_config,
                'layer_ranks': {**lrd_config.get('layer_ranks', {}), **layer_ranks},
            }

    target_net = build_target_net(cfg['target_net'], lrd_config)
    print(f'Target network: {cfg["target_net"]}, params: {target_net.get_total_params():,}')

    # Loss
    loss_fn = MappingLoss(sigma_noise=cfg.get('sigma_noise', 0.01)).to(device)

    experiment_name = make_experiment_name(cfg)
    checkpoint_dir = os.path.join(cfg['checkpoint_dir'], experiment_name)

    if cfg['training_strategy'] == 'slvt':
        mapping = build_generator(
            cfg.get('generator_type', 'linear'),
            target_net.get_total_params(),
            cfg['latent_dim'],
            cfg.get('alpha', 0.01),
            device,
        )
        print(f'Latent dim: {cfg["latent_dim"]}')
        print(f'Trainable: {mapping.trainable_params():,}')
        print(f'Fixed mapping weights: {mapping.W_fixed.numel():,}')

        trainer = SLVTTrainer(
            mapping,
            target_net,
            loss_fn,
            train_loader,
            test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            checkpoint_metadata={
                'target_net': cfg['target_net'],
                'training_strategy': 'slvt',
                'generator_type': cfg.get('generator_type', 'linear'),
                'latent_dim': cfg['latent_dim'],
                'alpha': cfg.get('alpha', 0.01),
                'sigma_noise': cfg.get('sigma_noise', 0.01),
                'lrd_config': cfg.get('lrd'),
            },
            save_interval=cfg.get('save_interval', 1),
            optimizer_name=cfg.get('optimizer', 'adamw'),
            scheduler_name=cfg.get('scheduler', 'cosine_annealing'),
        )
    elif cfg['training_strategy'] == 'lwt':
        trainer = LWTTrainer(
            target_net,
            loss_fn,
            cfg['layer_generators'],
            train_loader=train_loader,
            test_loader=test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            checkpoint_metadata={
                'target_net': cfg['target_net'],
                'training_strategy': 'lwt',
                'lrd_config': lrd_config,
                'sigma_noise': cfg.get('sigma_noise', 0.01),
            },
            save_interval=cfg.get('save_interval', 1),
            optimizer_name=cfg.get('optimizer', 'adamw'),
            scheduler_name=cfg.get('scheduler', 'cosine_annealing'),
        )
    else:
        raise ValueError(f'Unknown strategy: {cfg["training_strategy"]}')

    results = trainer.train()
    final_acc = results[-1]['test_acc']
    print(f'\nFinal test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
