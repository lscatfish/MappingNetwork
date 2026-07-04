"""
Train a baseline target network (without Mapping Network).

Usage:
  # 使用配置文件
  uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml

  # 使用命令行参数
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn1
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 1 --device cpu
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv

TARGET_NET_MAP = {
    'cnn2': CNN2, 'cnn1': CNN1, 'cnn1_3conv': CNN1_3Conv,
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., configs/cnn2_baseline.yaml)')
    parser.add_argument('--target', type=str, default=None,
                        choices=['cnn1', 'cnn2', 'cnn1_3conv'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # 如果给了配置文件，先读配置
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}

    # 命令行参数优先级高于配置文件
    target = args.target if args.target is not None else cfg.get('target')
    epochs = args.epochs if args.epochs is not None else cfg.get('epochs', 30)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 64)
    lr = args.lr if args.lr is not None else cfg.get('lr', 0.001)
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    device = args.device if args.device is not None else cfg.get('device', 'cuda')

    if target is None:
        parser.error('--target is required when no config file is provided')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TARGET_NET_MAP[target]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Training {target} baseline: {total_params:,} params')
    print(f'Device: {device}, Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = total = 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            _, pred = y_hat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, pred = y_hat.max(1)
                test_total += y.size(0)
                test_correct += pred.eq(y).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch}: test_acc={test_acc:.2f}%')

    checkpoint = {
        'type': 'baseline',
        'target_net': target,
        'epochs': epochs,
        'final_test_acc': test_acc,
        'state_dict': model.state_dict(),
    }
    save_path = f'{target}_baseline.pth'
    torch.save(checkpoint, save_path)
    print(f'Baseline saved to {save_path}')
    print(f'Final test accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
