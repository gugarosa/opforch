"""Data loading utilities returning PyTorch tensors."""

import json as j
from typing import Optional

import numpy as np
import torch

from opforch.utils import logging

logger = logging.get_logger(__name__)


def load_csv(csv_path: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
    """Loads a CSV file into a tensor.

    Args:
        csv_path: Path to the .csv file.
        device: Target device string.

    Returns:
        A tensor holding the loaded data, or None on failure.

    """

    logger.info("Loading file: %s ...", csv_path)

    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except OSError as err:
        logger.error(err)
        return None

    logger.info("File loaded.")

    tensor = torch.from_numpy(data).to(dtype=torch.float64)
    if device:
        tensor = tensor.to(device)

    return tensor


def load_txt(txt_path: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
    """Loads a .txt file into a tensor.

    Args:
        txt_path: Path to the .txt file.
        device: Target device string.

    Returns:
        A tensor holding the loaded data, or None on failure.

    """

    logger.info("Loading file: %s ...", txt_path)

    try:
        data = np.loadtxt(txt_path, delimiter=" ")
    except OSError as err:
        logger.error(err)
        return None

    logger.info("File loaded.")

    tensor = torch.from_numpy(data).to(dtype=torch.float64)
    if device:
        tensor = tensor.to(device)

    return tensor


def load_json(json_path: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
    """Loads a .json file into a tensor.

    Expects format: {"data": [{"id": ..., "label": ..., "features": [...]}, ...]}.

    Args:
        json_path: Path to the .json file.
        device: Target device string.

    Returns:
        A tensor holding the loaded data, or None on failure.

    """

    logger.info("Loading file: %s ...", json_path)

    try:
        with open(json_path) as f:
            json_file = j.load(f)
    except Exception as err:
        logger.error(err)
        return None

    logger.info("File loaded.")

    rows = []
    for d in json_file["data"]:
        meta = [d["id"], d["label"]]
        features = d["features"]
        rows.append(meta + features)

    tensor = torch.tensor(rows, dtype=torch.float64)
    if device:
        tensor = tensor.to(device)

    return tensor
