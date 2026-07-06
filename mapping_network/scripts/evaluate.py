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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get('batch_size', 64))

    checkpoint = torch.load(args.checkpoint, map_location=device)

    target_net = build_target_net(checkpoint['target_net'], checkpoint.get('lrd_config'))
    target_net = target_net.to(device)

    if checkpoint['training_strategy'] == 'slvt':
        mapping = build_generator(
            checkpoint.get('generator_type', 'linear'),
            target_net.get_total_params(),
            checkpoint['latent_dim'],
            checkpoint.get('alpha', 0.01),
            device,
            w_seed=checkpoint.get('w_seed', 12345),
        )
        mapping.load_state_dict(checkpoint['state_dict'], strict=False)
        theta_hat = mapping()
    elif checkpoint['training_strategy'] == 'lwt':
        # Rebuild layer mappings and load each state_dict
        layer_mappings = nn.ModuleDict()
        w_seed_base = checkpoint.get('w_seed', 12345)
        for idx, (name, gen_cfg) in enumerate(checkpoint['layer_generator_configs'].items()):
            group_size = target_net.get_group_param_size(name)
            mapping = build_generator(
                gen_cfg.get('type', 'linear'),
                group_size,
                gen_cfg['latent_dim'],
                gen_cfg.get('alpha', 0.01),
                device,
                w_seed=w_seed_base + idx,
            )
            mapping.load_state_dict(checkpoint['state_dict'][name], strict=False)
            layer_mappings[name] = mapping
        # Concatenate in the same order as target net param groups
        group_order = checkpoint.get('layer_group_order', list(layer_mappings.keys()))
        theta_hat = torch.cat([layer_mappings[name]() for name in group_order])
    else:
        raise ValueError(f'Unknown strategy: {checkpoint["training_strategy"]}')

    acc = evaluate_model(target_net, theta_hat, test_loader, device)
    print(f'Test accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
