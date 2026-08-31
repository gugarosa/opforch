"""Data loading utilities returning PyTorch tensors."""

from __future__ import annotations

import json

import numpy as np
import torch

from opforch.utils import logging

logger = logging.get_logger(__name__)


def _as_tensor(data, device: str | None) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float64, device=device)


def _load_text(path: str, delimiter: str, device: str | None) -> torch.Tensor | None:
    logger.info("Loading file: %s ...", path)
    try:
        data = np.loadtxt(path, delimiter=delimiter)
    except OSError as error:
        logger.error(error)
        return None
    logger.info("File loaded.")
    return _as_tensor(data, device)


def load_csv(csv_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load a CSV file into a tensor."""

    return _load_text(csv_path, ",", device)


def load_txt(txt_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load a whitespace-delimited text file into a tensor."""

    return _load_text(txt_path, " ", device)


def load_json(json_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load an OPF JSON file into a tensor."""

    logger.info("Loading file: %s ...", json_path)
    try:
        with open(json_path, encoding="utf-8") as file:
            records = json.load(file)["data"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.error(error)
        return None

    rows = [[record["id"], record["label"], *record["features"]] for record in records]
    logger.info("File loaded.")
    return _as_tensor(rows, device)
