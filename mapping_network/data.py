"""公共数据加载工具，供 train / evaluate / train_baseline 复用。"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST 标准化常量
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def get_mnist_loaders(
    batch_size: int = 64,
    root: str = './data',
    train: bool = True,
    download: bool = True,
):
    """返回 (train_loader, test_loader)。

    Args:
        batch_size: batch 大小。
        root: MNIST 数据集根目录。
        train: 是否返回 train_loader（False 时两个 loader 都指向 test 集，用于快速验证）。
        download: 是否自动下载。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])

    if train:
        train_dataset = datasets.MNIST(root, train=True, download=download, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    test_dataset = datasets.MNIST(root, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def get_mnist_test_loader(batch_size: int = 64, root: str = './data'):
    """仅返回 test_loader（用于 evaluate 脚本）。"""
    return get_mnist_loaders(batch_size, root, train=False)[1]
