# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Parse preloaded OPF tables into feature and label tensors.

"""  # fmt: skip

from __future__ import annotations

import torch

import opforch.utils.exception as e
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


def parse_loader(
    data: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Parse a preloaded OPF table into float32 features and int64 labels.

    Column 0 contains ignored sample IDs, column 1 contains labels, and columns 2 onward contain features.
    Non-tensor inputs are copied through torch.tensor with float64 dtype on PyTorch's default device.
    Tensor inputs keep their device, and slices may share input storage when no dtype conversion is needed.
    Labels are cast to int64 before validation, preserving truncation of fractional numeric labels.
    The unique-label count must equal the maximum converted label plus one, and a single label logs a warning.
    Tables must have at least one row and two columns. Two-column tables produce features of shape (N, 0).

    Args:
        data: Nonempty two-dimensional numeric tensor or array-like table with shape (N, F + 2).

    Returns:
        Float32 features of shape (N, F) and int64 labels of shape (N,), or (None, None) for malformed input.

    Raises:
        RuntimeError: PyTorch cannot execute a conversion or tensor operation, including backend and memory failures.

    """

    logger.info("Parsing data ...")

    try:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64)

        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
            raise e.ValueError(
                "`data` must be a nonempty two-dimensional table with ID and label columns, "
                f"but got shape {tuple(data.shape)}."
            )

        X = data[:, 2:].to(dtype=torch.float32)
        Y = data[:, 1].to(dtype=torch.int64)

        unique_labels = torch.unique(Y)
        n_unique = unique_labels.numel()

        if n_unique == 1:
            logger.warning("`labels.count=%s` indicates only one distinct label.", n_unique)
        if n_unique != (Y.max().item() + 1):
            raise e.ValueError("`labels` must be sequential values starting at zero after int64 conversion.")

        logger.info("Data parsed.")

        return X, Y

    except (TypeError, ValueError, IndexError, OverflowError, e.ValueError) as error:
        logger.error("`data_type=%s` could not be parsed (%s).", type(data).__name__, error)
        return None, None
