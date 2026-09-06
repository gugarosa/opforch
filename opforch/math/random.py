# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Generate random tensors with PyTorch's default generators.

Each call allocates a new tensor using PyTorch's default floating dtype, drawing from the generator for its device.
No seed is set internally, so callers control reproducibility through PyTorch's RNG state.

"""

from __future__ import annotations

import torch


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Draw uniformly distributed values using the selected device's default generator.

    PyTorch draws from [low, high) when low is less than high and returns the bound when they are equal.

    Args:
        low: Lower endpoint of the uniform interval.
        high: Upper endpoint of the uniform interval, greater than or equal to low.
        size: Number of values in the one-dimensional output tensor.
        device: Output device, or PyTorch's default device if None.

    Returns:
        torch.Tensor: (size,) tensor with PyTorch's default floating dtype on the selected device.

    Raises:
        RuntimeError: If low exceeds high or size is negative.

    """

    return torch.empty(size, device=device).uniform_(low, high)


def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Draw normally distributed values using the selected device's default generator.

    The legacy parameter named variance is passed directly as the standard deviation to ``Tensor.normal_``.
    It is not a statistical variance and is not square-rooted.

    Args:
        mean: Mean of the normal distribution.
        variance: Nonnegative standard deviation passed to PyTorch despite the legacy parameter name.
        size: Number of values in the one-dimensional output tensor.
        device: Output device, or PyTorch's default device if None.

    Returns:
        torch.Tensor: (size,) tensor with PyTorch's default floating dtype on the selected device.

    Raises:
        RuntimeError: If variance or size is negative.

    """

    return torch.empty(size, device=device).normal_(mean, variance)
