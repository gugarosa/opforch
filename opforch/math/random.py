"""Random number generators using PyTorch."""

from __future__ import annotations

import torch


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return uniformly distributed random values."""

    return torch.empty(size, device=device).uniform_(low, high)


def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return normally distributed random values."""

    return torch.empty(size, device=device).normal_(mean, variance)
