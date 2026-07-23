"""Baseline 训练入口（纯 torch，无 mapping）。

用法:
    uv run python3 examples/train_baseline.py --target cnn2 --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn1 import cnn1_baseline
from models.cnn1_3conv import cnn1_3conv_baseline
from models.cnn2 import cnn2_baseline

from data import get_mnist_loaders

MODEL_FACTORIES = {
    'cnn2': cnn2_baseline,
    'cnn1': cnn1_baseline,
    'cnn1_3conv': cnn1_3conv_baseline,
}


def main():
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--target', choices=['cnn1', 'cnn2', 'cnn1_3conv'], default='cnn2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    experiment_name = f'{args.target}_baseline'
    ckpt_dir = Path(args.checkpoint_dir) / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    net = MODEL_FACTORIES[args.target]().to(args.device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = -1.0
    results = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            logits = net(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100*correct/total:.1f}%')

        scheduler.step()
        train_acc = 100.0 * correct / total

        net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                _, pred = net(x).max(1)
                test_total += y.size(0)
                test_correct += pred.eq(y).sum().item()
        test_acc = 100.0 * test_correct / test_total

        results.append({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_acc': train_acc,
            'test_acc': test_acc,
        })
        print(f'Epoch {epoch}: train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), ckpt_dir / f'{experiment_name}_best.pth')

    torch.save(net.state_dict(), ckpt_dir / f'{experiment_name}_final.pth')
    with open(ckpt_dir / f'{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nDone. Best test acc: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
