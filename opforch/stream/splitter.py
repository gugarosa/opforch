"""Data splitting utilities using PyTorch tensor operations."""

from typing import Tuple

import torch

import opforch.utils.exception as e
from opforch.utils import logging

logger = logging.get_logger(__name__)


def split(
    X: torch.Tensor,
    Y: torch.Tensor,
    percentage: float = 0.5,
    random_state: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits data into two new sets via random permutation.

    Args:
        X: Feature tensor of shape (N, D).
        Y: Label tensor of shape (N,).
        percentage: Fraction of data in the first set.
        random_state: Random seed for reproducibility.

    Returns:
        (X_1, X_2, Y_1, Y_2) split tensors.

    """

    logger.info("Splitting data ...")

    torch.manual_seed(random_state)

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` and `Y` should have the same number of samples")

    idx = torch.randperm(X.shape[0])
    halt = int(len(X) * percentage)

    X_1, X_2 = X[idx[:halt]], X[idx[halt:]]
    Y_1, Y_2 = Y[idx[:halt]], Y[idx[halt:]]

    logger.debug(
        "X_1: %s | X_2: %s | Y_1: %s | Y_2: %s.",
        X_1.shape, X_2.shape, Y_1.shape, Y_2.shape,
    )
    logger.info("Data split.")

    return X_1, X_2, Y_1, Y_2


def split_with_index(
    X: torch.Tensor,
    Y: torch.Tensor,
    percentage: float = 0.5,
    random_state: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits data into two new sets, also returning original indices.

    Args:
        X: Feature tensor of shape (N, D).
        Y: Label tensor of shape (N,).
        percentage: Fraction of data in the first set.
        random_state: Random seed for reproducibility.

    Returns:
        (X_1, X_2, Y_1, Y_2, I_1, I_2) split tensors with index tensors.

    """

    logger.info("Splitting data ...")

    torch.manual_seed(random_state)

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` and `Y` should have the same number of samples")

    idx = torch.randperm(X.shape[0])
    halt = int(len(X) * percentage)

    I_1, I_2 = idx[:halt], idx[halt:]
    X_1, X_2 = X[I_1], X[I_2]
    Y_1, Y_2 = Y[I_1], Y[I_2]

    logger.debug(
        "X_1: %s | X_2: %s | Y_1: %s | Y_2: %s.",
        X_1.shape, X_2.shape, Y_1.shape, Y_2.shape,
    )
    logger.info("Data split.")

    return X_1, X_2, Y_1, Y_2, I_1, I_2


def merge(
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    Y_1: torch.Tensor,
    Y_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merges two sets into one.

    Args:
        X_1: First feature tensor.
        X_2: Second feature tensor.
        Y_1: First label tensor.
        Y_2: Second label tensor.

    Returns:
        (X, Y) merged tensors.

    """

    logger.info("Merging data ...")

    X = torch.cat([X_1, X_2], dim=0)
    Y = torch.cat([Y_1, Y_2], dim=0)

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError(
            "`(X_1, X_2)` and `(Y_1, Y_2)` should have the same number of samples"
        )

    logger.debug("X: %s | Y: %s.", X.shape, Y.shape)
    logger.info("Data merged.")

    return X, Y
