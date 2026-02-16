"""Random number generators using PyTorch."""

from typing import Optional

import torch


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates random numbers from a uniform distribution.

    Args:
        low: Lower interval bound.
        high: Upper interval bound.
        size: Number of values to generate.
        device: Target device.

    Returns:
        Tensor of uniform random values.

    """

    return torch.empty(size, device=device).uniform_(low, high)


def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates random numbers from a Gaussian distribution.

    Args:
        mean: Gaussian mean.
        variance: Gaussian standard deviation.
        size: Number of values to generate.
        device: Target device.

    Returns:
        Tensor of Gaussian random values.

    """

    return torch.empty(size, device=device).normal_(mean, variance)
