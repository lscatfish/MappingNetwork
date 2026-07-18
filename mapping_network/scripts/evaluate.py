"""
Evaluate a trained Mapping Network checkpoint.

Supports both SLVT (single ParameterGenerator) and LWT (per-layer ParameterGenerators).

Usage:
  # SLVT checkpoint
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth \\
      --config configs/cnn2_slvt.yaml

  # LWT checkpoint (dict of per-layer state_dicts)
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_lwt/cnn2_lwt_final.pth \\
      --config configs/cnn2_lwt.yaml
"""

import argparse

import torch
import yaml

from mapping_network.data import get_mnist_test_loader
from mapping_network.factory import build_generator, build_target_net


def evaluate_model(target_net, theta_hat, test_loader, device):
    """Run one evaluation pass using a full generated parameter vector."""
    target_net.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = target_net.functional_forward(x, theta_hat)
        _, pred = y_hat.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = (
        args.device
        if args.device
        else cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

    test_loader = get_mnist_test_loader(cfg.get('batch_size', 64), root='./data')

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    target_net = build_target_net(checkpoint['target_net'], checkpoint.get('lrd_config'))
    target_net = target_net.to(device)

    if checkpoint['training_strategy'] == 'slvt':
        # 从 checkpoint 恢复 gen_config
        gen_config = checkpoint.get('gen_config')
        if gen_config is None:
            # 兼容旧 checkpoint（无 gen_config 字段）
            gen_config = {
                'type': checkpoint.get('generator_type', 'linear'),
                'latent_dim': checkpoint['latent_dim'],
                'alpha': checkpoint.get('alpha', 0.01),
            }
            # 仅在 checkpoint 显式保存了 w_seed 时透传，否则让 generator 自行派生
            if 'w_seed' in checkpoint:
                gen_config['w_seed'] = checkpoint['w_seed']
        mapping = build_generator(gen_config, target_net.get_total_params(), device=device)
        mapping.load_persistent_state_dict(checkpoint['state_dict'])
        theta_hat = mapping()
    elif checkpoint['training_strategy'] == 'lwt':
        # Rebuild layer mappings and load each state_dict
        layer_mappings = torch.nn.ModuleDict()
        for name, gen_cfg in checkpoint['layer_generator_configs'].items():
            group_size = target_net.get_group_param_size(name)
            config = dict(gen_cfg)
            config['layer_name'] = name
            mapping = build_generator(config, target_total_params=group_size, device=device)
            mapping.load_persistent_state_dict(checkpoint['state_dict'][name])
            layer_mappings[name] = mapping
        # Concatenate in the same order as target net param groups
        group_order = checkpoint.get('layer_group_order', list(layer_mappings.keys()))
        group_theta = {name: layer_mappings[name]() for name in group_order}
        theta_hat = target_net.assemble_params(group_theta)
    else:
        raise ValueError(f'Unknown strategy: {checkpoint["training_strategy"]}')

    acc = evaluate_model(target_net, theta_hat, test_loader, device)
    print(f'Test accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
