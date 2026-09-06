# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Load numeric data into PyTorch tensors without changing IDs or labels.

"""  # fmt: skip

from __future__ import annotations

import json

import numpy as np
import torch

from opforch.utils.logging import get_logger

logger = get_logger(__name__)


def _as_tensor(data, device: str | None) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float64, device=device)


def _load_text(path: str, delimiter: str, device: str | None) -> torch.Tensor | None:
    logger.info("Loading file: %s ...", path)
    try:
        data = np.loadtxt(path, delimiter=delimiter)
    except OSError as error:
        logger.error("`path=%s` could not be loaded (%s).", path, error)
        return None

    logger.info("File loaded.")
    return _as_tensor(data, device)


def load_csv(csv_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load a comma-delimited numeric file into a float64 tensor.

    Parsing follows numpy.loadtxt, including its dimension squeezing. A regular table has shape (N, C),
    a single row or column becomes one-dimensional, and a single value becomes scalar.
    An empty file yields shape (0,). No ID or label conversion is performed.
    CPU tensors may share the NumPy allocation through torch.as_tensor rather than making a copy.
    Device selection follows PyTorch's default allocation policy without automatic CUDA discovery.

    Args:
        csv_path: Path to a comma-delimited numeric file, read without modification.
        device: Device passed to torch.as_tensor, or PyTorch's default device when device is None.

    Returns:
        A float64 tensor with numpy.loadtxt's shape, or None if the file cannot be opened or read.

    Raises:
        ValueError: Numeric text is malformed or rows have inconsistent widths.
        RuntimeError: PyTorch cannot create the tensor on the requested device.

    """

    return _load_text(csv_path, ",", device)


def load_txt(txt_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load a space-delimited numeric file into a float64 tensor.

    Parsing follows numpy.loadtxt with a literal space delimiter, not arbitrary whitespace.
    A regular table has shape (N, C), a single row or column becomes one-dimensional, and a single value is scalar.
    An empty file yields shape (0,). No ID or label conversion is performed.
    CPU tensors may share the NumPy allocation through torch.as_tensor rather than making a copy.
    Device selection follows PyTorch's default allocation policy without automatic CUDA discovery.

    Args:
        txt_path: Path to a space-delimited numeric file, read without modification.
        device: Device passed to torch.as_tensor, or PyTorch's default device when device is None.

    Returns:
        A float64 tensor with numpy.loadtxt's shape, or None if the file cannot be opened or read.

    Raises:
        ValueError: Numeric text is malformed or rows have inconsistent widths.
        RuntimeError: PyTorch cannot create the tensor on the requested device.

    """

    return _load_text(txt_path, " ", device)


def load_json(json_path: str, device: str | None = None) -> torch.Tensor | None:
    """Load a UTF-8 OPF JSON data collection into a float64 tensor.

    The top-level data list contains records with id, label, and features fields.
    Each row is ``[id, label, *features]``, with IDs and labels left unchanged.
    Nonempty rectangular records produce shape (N, F + 2), including the row axis for a single record.
    An empty data list produces shape (0,). Numeric records are copied into a new tensor.
    I/O, JSON decoding, and top-level data lookup errors return None, while record and tensor errors propagate.
    Device selection follows PyTorch's default allocation policy without automatic CUDA discovery.

    Args:
        json_path: Path to a UTF-8 JSON file containing the data list, read without modification.
        device: Device passed to torch.as_tensor, or PyTorch's default device when device is None.

    Returns:
        A float64 tensor of record rows, or None on I/O, decoding, or top-level data lookup failures.

    Raises:
        KeyError: A record lacks an id, label, or features field.
        TypeError: Records or feature collections are not iterable in the expected form.
        ValueError: Record values cannot form a rectangular numeric tensor.
        RuntimeError: PyTorch cannot create the tensor on the requested device.

    """

    logger.info("Loading file: %s ...", json_path)
    try:
        with open(json_path, encoding="utf-8") as file:
            records = json.load(file)["data"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.error("`json_path=%s` could not be loaded (%s).", json_path, error)
        return None

    rows = [[record["id"], record["label"], *record["features"]] for record in records]
    logger.info("File loaded.")
    return _as_tensor(rows, device)
