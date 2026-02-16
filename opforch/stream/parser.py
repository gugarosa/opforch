"""Data parsing utilities for OPF file format."""

from typing import Optional, Tuple

import torch

import opforch.utils.exception as e
from opforch.utils import logging

logger = logging.get_logger(__name__)


def parse_loader(
    data: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Parses data in OPF file format that was pre-loaded.

    OPF format: column 0 = sample id, column 1 = label, columns 2+ = features.

    Args:
        data: Tensor holding the loaded data.

    Returns:
        Tuple of (features, labels) tensors, or (None, None) on failure.

    """

    logger.info("Parsing data ...")

    try:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64)

        X = data[:, 2:].to(dtype=torch.float32)
        Y = data[:, 1].to(dtype=torch.int64)

        unique_labels = torch.unique(Y)
        n_unique = unique_labels.numel()

        if n_unique == 1:
            logger.warning("Parsed data only have a single label.")
        if n_unique != (Y.max().item() + 1):
            raise e.ValueError(
                "Parsed data should have sequential labels, e.g., 0, 1, ..., n-1"
            )

        logger.info("Data parsed.")

        return X, Y

    except Exception as error:
        logger.error(error)
        return None, None
