# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Split and merge datasets using PyTorch tensor operations.

"""  # fmt: skip

from __future__ import annotations

import torch

import opforch.utils.exception as e
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


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
    """Shuffle aligned samples into two partitions and return their original indices.

    A single permutation is sliced at int(N * percentage), with ordinary PyTorch slicing behavior.
    Advanced indexing creates new feature and label tensors with their original dtypes, devices, and trailing shapes.
    Indices have int64 dtype and use torch.randperm's default allocation device.
    Calling torch.manual_seed resets the global PyTorch RNG state before generating the permutation.

    Args:
        X: Feature tensor with shape (N, ...) and a sample axis aligned with Y.
        Y: Label tensor with shape (N, ...) and a sample axis aligned with X.
        percentage: Multiplier used for int(N * percentage), the permutation slice boundary.
        random_state: Seed passed to torch.manual_seed before shuffling.

    Returns:
        X_first, X_second, Y_first, Y_second, indices_first, indices_second, in that order.

    Raises:
        SizeError: X and Y have different sample counts.
        RuntimeError: PyTorch cannot initialize the RNG or index the input tensors.

    """

    logger.info("Splitting data ...")

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` must have the same number of samples as Y.")

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
    """Shuffle aligned samples into two feature and label partitions.

    The permutation is sliced at int(N * percentage) with ordinary PyTorch slicing behavior.
    Outputs are new tensors retaining input dtypes, devices, and trailing shapes.
    As in split_with_index, torch.manual_seed resets the global PyTorch RNG state before shuffling.

    Args:
        X: Feature tensor with shape (N, ...) and a sample axis aligned with Y.
        Y: Label tensor with shape (N, ...) and a sample axis aligned with X.
        percentage: Multiplier used for int(N * percentage), the permutation slice boundary.
        random_state: Seed passed to torch.manual_seed before shuffling.

    Returns:
        X_first, X_second, Y_first, Y_second, in that order.

    Raises:
        SizeError: X and Y have different sample counts.
        RuntimeError: PyTorch cannot initialize the RNG or index the input tensors.

    """

    return split_with_index(X, Y, percentage, random_state)[:4]


def merge(
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    Y_1: torch.Tensor,
    Y_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two aligned feature and label partitions in their given order.

    Inputs are left unchanged. Concatenation preserves compatible trailing dimensions and follows
    PyTorch's dtype promotion and device compatibility rules. The RNG state is not changed.

    Args:
        X_1: First feature partition with shape (N_1, ...).
        X_2: Second feature partition with shape (N_2, ...) and trailing dimensions compatible with X_1.
        Y_1: First label partition with shape (N_1, ...).
        Y_2: Second label partition with shape (N_2, ...) and trailing dimensions compatible with Y_1.

    Returns:
        Merged features followed by merged labels, each with N_1 + N_2 rows.

    Raises:
        SizeError: Either feature partition has a different sample count from its corresponding labels.
        RuntimeError: PyTorch cannot concatenate incompatible shapes or devices.

    """

    logger.info("Merging data ...")

    if X_1.shape[0] != Y_1.shape[0]:
        raise e.SizeError("`X_1` must have the same number of samples as Y_1.")
    if X_2.shape[0] != Y_2.shape[0]:
        raise e.SizeError("`X_2` must have the same number of samples as Y_2.")

    X = torch.cat((X_1, X_2))
    Y = torch.cat((Y_1, Y_2))

    logger.info("Data merged.")
    return X, Y
