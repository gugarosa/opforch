"""Convert OPF binary data to text formats."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from opforch.utils import logging

logger = logging.get_logger(__name__)


def _read_opf_binary(opf_path: str) -> list[tuple]:
    with open(opf_path, "rb") as file:
        n_samples, _, n_features = struct.unpack("<iii", file.read(12))
        record = struct.Struct(f"<ii{'f' * n_features}")
        return [
            (sample_id, label - 1, *features)
            for sample_id, label, *features in (
                record.unpack(file.read(record.size)) for _ in range(n_samples)
            )
        ]


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
    """Convert a binary OPF file to text."""

    _save_table(opf_path, output_file, ".txt", " ")


def opf2csv(opf_path: str, output_file: str | None = None) -> None:
    """Convert a binary OPF file to CSV."""

    _save_table(opf_path, output_file, ".csv", ",")


def opf2json(opf_path: str, output_file: str | None = None) -> None:
    """Convert a binary OPF file to JSON."""

    output_file = _output_path(opf_path, output_file, ".json")
    records = [
        {"id": sample_id, "label": label, "features": list(features)}
        for sample_id, label, *features in _read_opf_binary(opf_path)
    ]
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump({"data": records}, file)
    logger.info("File converted to %s.", output_file)
