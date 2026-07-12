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

from mapping_network.data import get_mnist_loaders
from mapping_network.factory import build_generator, build_target_net
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _merge_lwt_lrd_config(cfg: dict, lrd_config: dict) -> dict:
    """Merge per-layer LRD overrides from LWT layer_generators into global LRDConfig."""
    if cfg.get('training_strategy') != 'lwt':
        return lrd_config
    layer_ranks = {}
    layer_enabled = {}
    for name, gen_cfg in cfg['layer_generators'].items():
        if 'lrd_rank' in gen_cfg:
            layer_ranks[name] = gen_cfg['lrd_rank']
        if 'lrd_enabled' in gen_cfg:
            layer_enabled[name] = gen_cfg['lrd_enabled']
    return {
        **lrd_config,
        'layer_ranks': {**lrd_config.get('layer_ranks', {}), **layer_ranks},
        'layer_enabled': {**lrd_config.get('layer_enabled', {}), **layer_enabled},
    }


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
    parser.add_argument('--resume', type=str, default=None)
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
    train_loader, test_loader = get_mnist_loaders(cfg['batch_size'], root='./data')

    lrd_config = cfg.get('lrd', {})

    lrd_config = _merge_lwt_lrd_config(cfg, lrd_config)

    target_net = build_target_net(cfg['target_net'], lrd_config)
    print(f'Target network: {cfg["target_net"]}, params: {target_net.get_total_params():,}')

    # Loss
    loss_fn = MappingLoss(
        sigma_noise=cfg.get('sigma_noise', 0.0001),
        lambda_st_init=cfg.get('lambda_st_init', 0.1),
        lambda_sm_init=cfg.get('lambda_sm_init', 0.1),
        lambda_al_init=cfg.get('lambda_al_init', 0.1),
        n_stab_samples=cfg.get('n_stab_samples', 5),
    ).to(device)

    experiment_name = make_experiment_name(cfg)
    checkpoint_dir = os.path.join(cfg['checkpoint_dir'], experiment_name)

    append_log = args.resume is not None

    if cfg['training_strategy'] == 'slvt':
        generator_type = cfg.get('generator_type', 'linear')

        # 构建 generator_config dict —— 只透传 generator 需要的配置，
        # factory 负责注入 target_total_params 和 device，不再硬编码 w_seed
        gen_config = {
            'type': generator_type,
            'latent_dim': cfg['latent_dim'],
            'alpha': cfg.get('alpha', 0.01),
        }
        # 可选：如果用户显式指定了 w_seed，透传给 generator（由 generator 内部管理）
        if 'w_seed' in cfg:
            gen_config['w_seed'] = cfg['w_seed']

        mapping = build_generator(gen_config, target_net.get_total_params(), device=device)
        print(f'Generator: {generator_type}')
        print(f'Latent dim: {cfg["latent_dim"]}')
        print(f'Trainable: {mapping.trainable_params():,}')
        print(f'Fixed mapping weights: {mapping.fixed_params_count():,}')

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
                'gen_config': gen_config,
                'latent_dim': cfg['latent_dim'],
                'alpha': cfg.get('alpha', 0.01),
                'sigma_noise': cfg.get('sigma_noise', 0.0001),
                'lrd_config': cfg.get('lrd'),
                'warmup_epochs': cfg.get('warmup_epochs', max(1, cfg['epochs'] // 10)),
            },
            save_interval=cfg.get('save_interval', 1),
            optimizer_name=cfg.get('optimizer', 'adamw'),
            scheduler_name=cfg.get('scheduler', 'cosine_annealing'),
            append_log=append_log,
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
                'sigma_noise': cfg.get('sigma_noise', 0.0001),
                'warmup_epochs': cfg.get('warmup_epochs', max(1, cfg['epochs'] // 10)),
            },
            save_interval=cfg.get('save_interval', 1),
            optimizer_name=cfg.get('optimizer', 'adamw'),
            scheduler_name=cfg.get('scheduler', 'cosine_annealing'),
            append_log=append_log,
        )
    else:
        raise ValueError(f'Unknown strategy: {cfg["training_strategy"]}')

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    else:
        start_epoch = 1

    results = trainer.train(start_epoch=start_epoch)
    final_acc = results[-1]['test_acc']
    print(f'\nFinal test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
