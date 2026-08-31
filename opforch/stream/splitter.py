"""Data splitting utilities using PyTorch tensor operations."""

from __future__ import annotations

import torch

import opforch.utils.exception as e
from opforch.utils import logging

logger = logging.get_logger(__name__)


def split_with_index(
    X: torch.Tensor,
    Y: torch.Tensor,
    percentage: float = 0.5,
    random_state: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split data and return both partitions with their original indices."""

    logger.info("Splitting data ...")

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` and `Y` should have the same number of samples")

    torch.manual_seed(random_state)
    indices = torch.randperm(X.shape[0])
    halt = int(len(X) * percentage)
    first, second = indices[:halt], indices[halt:]
    result = X[first], X[second], Y[first], Y[second], first, second
    logger.info("Data split.")
    return result


def split(
    X: torch.Tensor,
    Y: torch.Tensor,
    percentage: float = 0.5,
    random_state: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into two partitions."""

    return split_with_index(X, Y, percentage, random_state)[:4]


def merge(
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    Y_1: torch.Tensor,
    Y_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two feature and label partitions."""

    logger.info("Merging data ...")
    X = torch.cat((X_1, X_2))
    Y = torch.cat((Y_1, Y_2))
    if X.shape[0] != Y.shape[0]:
        raise e.SizeError(
            "`(X_1, X_2)` and `(Y_1, Y_2)` should have the same number of samples"
        )
    logger.info("Data merged.")
    return X, Y
