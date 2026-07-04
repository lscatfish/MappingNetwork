"""Shared test fixtures and configurations."""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        '--device',
        action='store',
        default=None,
        help='Device to run tests on: "cuda", "cpu", or None (auto-detect)',
    )


@pytest.fixture(scope='session')
def device(request):
    """Return device for tests. Uses CUDA if available by default."""
    opt = request.config.getoption('--device')
    if opt is not None:
        return opt
    return 'cuda' if torch.cuda.is_available() else 'cpu'
