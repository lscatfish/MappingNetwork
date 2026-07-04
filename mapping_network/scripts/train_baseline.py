"""
Train a baseline target network (without Mapping Network).

Usage:
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn1
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 1 --device cpu
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True,
                        choices=['cnn1', 'cnn2', 'cnn1_3conv'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = TARGET_NET_MAP[args.target]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Training {args.target} baseline: {total_params:,} params')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
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
        'target_net': args.target,
        'epochs': args.epochs,
        'final_test_acc': test_acc,
        'state_dict': model.state_dict(),
    }
    save_path = f'{args.target}_baseline.pth'
    torch.save(checkpoint, save_path)
    print(f'Baseline saved to {save_path}')
    print(f'Final test accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
