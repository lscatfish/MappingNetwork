"""
Evaluate a trained Mapping Network checkpoint.

Supports both SLVT (single MappingNetwork) and LWT (per-layer MappingNetworks).

Usage:
  # SLVT checkpoint
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_slvt_final.pth \\
      --config configs/cnn2_slvt.yaml

  # LWT checkpoint (dict of per-layer state_dicts)
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_lwt_final.pth \\
      --config configs/cnn2_lwt.yaml
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv
from mapping_network.mapping.mapping_net import MappingNetwork

TARGET_NET_MAP = {
    'cnn2': CNN2, 'cnn1': CNN1, 'cnn1_3conv': CNN1_3Conv,
}


def build_param_groups(target_net):
    """Group target net params by layer name prefix, matching LWTTrainer logic."""
    groups = {}
    for name, param in target_net.named_parameters():
        base = name.split('.')[0]
        if base not in groups:
            groups[base] = 0
        groups[base] += param.numel()
    return list(groups.items())


@torch.no_grad()
def evaluate_slvt(mapping, target_net, test_loader, device):
    mapping.eval()
    target_net.eval()
    theta = mapping()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = target_net.functional_forward(x, theta)
        _, pred = y_hat.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return 100. * correct / total


@torch.no_grad()
def evaluate_lwt(mappings, param_groups, target_net, test_loader, device):
    """LWT evaluation: concatenate per-layer mapping outputs into full theta_hat."""
    target_net.eval()
    for mapping in mappings.values():
        mapping.eval()

    # Concatenate per-group theta in the same order as param_groups
    all_theta = [mappings[name]() for name, _ in param_groups]
    theta_hat = torch.cat(all_theta)

    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = target_net.functional_forward(x, theta_hat)
        _, pred = y_hat.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if args.device else cfg.get(
        'device', 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    target_cls = TARGET_NET_MAP[cfg['target_net']]
    target_net = target_cls().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get('batch_size', 64))

    if cfg.get('training_strategy') == 'lwt':
        # LWT: recreate per-layer MappingNetworks matching the trainer's group sizes
        param_groups = build_param_groups(target_net)
        mappings = {}
        layer_dims = cfg.get('layer_latent_dims', {})
        for name, group_size in param_groups:
            dim = layer_dims.get(name, 64)
            mappings[name] = MappingNetwork(
                group_size, dim, alpha=cfg.get('alpha', 0.01)
            ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        for name, mapping in mappings.items():
            if name in checkpoint:
                mapping.load_state_dict(checkpoint[name])

        acc = evaluate_lwt(mappings, param_groups, target_net, test_loader, device)
    else:
        # SLVT: single MappingNetwork
        mapping = MappingNetwork(
            target_net.get_total_params(),
            cfg.get('latent_dim', 2048),
            alpha=cfg.get('alpha', 0.01),
        ).to(device)
        mapping.load_state_dict(torch.load(args.checkpoint, map_location=device))
        acc = evaluate_slvt(mapping, target_net, test_loader, device)

    print(f'Test accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
