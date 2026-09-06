# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Convert little-endian OPF binary records to text formats.

The binary header contains three signed 32-bit integers: sample count, label count, and feature count.
Each record contains signed 32-bit sample ID and label fields followed by float32 features.
The header's label count is read but not validated. Every label is shifted by -1 without remapping IDs.
Exactly the declared sample count is read, and trailing bytes are ignored.

"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from opforch.utils.logging import get_logger

logger = get_logger(__name__)


def _read_opf_binary(opf_path: str) -> list[tuple[int | float, ...]]:
    with open(opf_path, "rb") as file:
        n_samples, _, n_features = struct.unpack("<iii", file.read(12))
        record = struct.Struct(f"<ii{'f' * n_features}")
        rows = []
        for _ in range(n_samples):
            sample_id, label, *features = record.unpack(file.read(record.size))
            rows.append((sample_id, label - 1, *features))

    return rows


def _output_path(source: str, output: str | None, suffix: str) -> str:
    return output or str(Path(source).with_suffix(suffix))


def _save_table(
    opf_path: str,
    output_file: str | None,
    suffix: str,
    delimiter: str,
) -> None:
    output_file = _output_path(opf_path, output_file, suffix)
    np.savetxt(output_file, _read_opf_binary(opf_path), delimiter=delimiter)
    logger.info("File converted to %s.", output_file)


def opf2txt(opf_path: str, output_file: str | None = None) -> None:
    """Write little-endian OPF records as a space-delimited numeric table.

    Output rows contain sample ID, label minus one, and the float32 feature values decoded from the binary file.
    NumPy's savetxt formatting is used without a header, overwriting any existing destination.
    Choosing the source as the destination overwrites it after all binary records have been read.
    No parent directories are created, and read or write failures propagate.

    Args:
        opf_path: Path to the binary OPF source, fully read before the destination is opened.
        output_file: Destination path, or None or an empty string to replace the source suffix with .txt.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: The binary header or a declared record is incomplete.

    """

    _save_table(opf_path, output_file, ".txt", " ")


def opf2csv(opf_path: str, output_file: str | None = None) -> None:
    """Write little-endian OPF records as a comma-delimited numeric table.

    Output rows contain sample ID, label minus one, and the float32 feature values decoded from the binary file.
    NumPy's savetxt formatting is used without a header, overwriting any existing destination.
    Choosing the source as the destination overwrites it after all binary records have been read.
    No parent directories are created, and read or write failures propagate.

    Args:
        opf_path: Path to the binary OPF source, fully read before the destination is opened.
        output_file: Destination path, or None or an empty string to replace the source suffix with .csv.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: The binary header or a declared record is incomplete.

    """

    _save_table(opf_path, output_file, ".csv", ",")


def opf2json(opf_path: str, output_file: str | None = None) -> None:
    """Write little-endian OPF records as a UTF-8 JSON data collection.

    The top-level data list contains objects with integer id and label fields and a features list.
    Labels are shifted by -1, and binary float32 features are decoded as Python floats.
    Any existing destination is overwritten, no parent directories are created, and I/O failures propagate.
    Choosing the source as the destination overwrites it after all binary records have been read.

    Args:
        opf_path: Path to the binary OPF source, fully read before the destination is opened.
        output_file: Destination path, or None or an empty string to replace the source suffix with .json.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: The binary header or a declared record is incomplete.

    """

    output_file = _output_path(opf_path, output_file, ".json")
    records = [
        {"id": sample_id, "label": label, "features": list(features)}
        for sample_id, label, *features in _read_opf_binary(opf_path)
    ]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump({"data": records}, file)

    logger.info("File converted to %s.", output_file)
