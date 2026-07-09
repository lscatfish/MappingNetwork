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
        w_seed = cfg.get('w_seed', 12345)
        generator_type = cfg.get('generator_type', 'linear')

        # 构建 generator_config dict
        gen_config = {
            'type': generator_type,
            'latent_dim': cfg['latent_dim'],
            'alpha': cfg.get('alpha', 0.01),
            'w_seed': w_seed,
        }

        # 生成器特定参数从 cfg 直接透传
        if generator_type == 'kron_structured':
            gen_config['d1'] = cfg.get('d1')
            gen_config['d2'] = cfg.get('d2')
        elif generator_type == 'kron_weight':
            gen_config['rank'] = cfg.get('kron_rank', 8)
            # layer_shapes 从 target_net 的 param slices 构建
            slices = target_net.get_param_slices()
            layer_shapes = {}
            for s in slices:
                if s.kind == 'full':
                    name = s.name
                    if s.is_bias:
                        layer_shapes[name] = (s.shape[0], 1, True)
                    else:
                        m = s.shape[0]
                        n = 1
                        for dim in s.shape[1:]:
                            n *= dim
                        layer_shapes[name] = (m, n, False)
                elif s.kind == 'lrd':
                    layer_shapes[s.weight_name] = (s.u_shape[0], s.v_shape[0], False)
                    layer_shapes[s.bias_name] = (s.b_shape[0], 1, True)
            gen_config['layer_shapes'] = layer_shapes
        elif generator_type == 'pca':
            trajectory_path = cfg.get('weight_trajectory_path')
            if trajectory_path:
                data = torch.load(trajectory_path, map_location='cpu', weights_only=False)
                trajectory = data['trajectory']
                from sklearn.decomposition import PCA
                pca = PCA(n_components=cfg['latent_dim'])
                pca.fit(trajectory.numpy())
                gen_config['pca_components'] = torch.tensor(
                    pca.components_, dtype=torch.float32
                )
                gen_config['pca_mean'] = torch.tensor(
                    pca.mean_, dtype=torch.float32
                )
        elif generator_type == 'adaptive_dim':
            gen_config['gate_init'] = cfg.get('gate_init', 0.0)
            gen_config['active_dim'] = cfg.get('active_dim')
        elif generator_type == 'manifold_reg':
            gen_config['n_geodesic_samples'] = cfg.get('n_geodesic_samples', 5)
            gen_config['geodesic_eps'] = cfg.get('geodesic_eps', 1e-3)
            gen_config['ib_beta'] = cfg.get('ib_beta', 0.001)
            anchor_path = cfg.get('anchor_checkpoint')
            if anchor_path:
                ckpt = torch.load(anchor_path, map_location='cpu', weights_only=False)
                gen_config['theta_anchor'] = ckpt.get('theta_anchor')
        elif generator_type == 'superposition':
            gen_config['num_tasks'] = cfg.get('num_tasks', 1)
            gen_config['task_id'] = cfg.get('task_id', 0)
        elif generator_type == 'tt_structured':
            gen_config['tt_shape'] = cfg.get('tt_shape', 2)
            gen_config['tt_rank'] = cfg.get('tt_rank', 2)

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
                'generator_type': generator_type,
                'gen_config': gen_config,
                'latent_dim': cfg['latent_dim'],
                'alpha': cfg.get('alpha', 0.01),
                'sigma_noise': cfg.get('sigma_noise', 0.0001),
                'lrd_config': cfg.get('lrd'),
                'w_seed': w_seed,
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
                'w_seed': cfg.get('w_seed', 12345),
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
