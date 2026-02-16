"""Converts OPF binary data to a variety of extensions."""

import json as j
import struct
from typing import Optional

import numpy as np
import torch

from opforch.utils import logging

logger = logging.get_logger(__name__)


def _read_opf_binary(opf_path: str):
    """Reads a binary OPF file and returns raw sample tuples.

    Args:
        opf_path: Path to the binary .dat or .opf file.

    Returns:
        A list of sample tuples (id, label-1, features...).

    """

    header_format = "<iii"
    header_size = struct.calcsize(header_format)

    with open(opf_path, "rb") as f:
        header_data = struct.unpack(header_format, f.read(header_size))

        n_samples = header_data[0]
        n_features = header_data[2]

        file_format = "<ii"
        for _ in range(n_features):
            file_format += "f"

        data_size = struct.calcsize(file_format)

        samples = []
        for _ in range(n_samples):
            data = struct.unpack(file_format, f.read(data_size))
            samples.append((data[0], data[1] - 1, *data[2:]))

    return samples


def opf2txt(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .txt file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    samples = _read_opf_binary(opf_path)

    if not output_file:
        output_file = opf_path.split(".")[0] + ".txt"

    np.savetxt(output_file, samples, delimiter=" ")

    logger.info("File converted to %s.", output_file)


def opf2csv(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .csv file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    samples = _read_opf_binary(opf_path)

    if not output_file:
        output_file = opf_path.split(".")[0] + ".csv"

    np.savetxt(output_file, samples, delimiter=",")

    logger.info("File converted to %s.", output_file)


def opf2json(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .json file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    header_format = "<iii"
    header_size = struct.calcsize(header_format)

    with open(opf_path, "rb") as f:
        header_data = struct.unpack(header_format, f.read(header_size))

        n_samples = header_data[0]
        n_features = header_data[2]

        file_format = "<ii"
        for _ in range(n_features):
            file_format += "f"

        data_size = struct.calcsize(file_format)

        json_data = {"data": []}
        for _ in range(n_samples):
            data = struct.unpack(file_format, f.read(data_size))
            json_data["data"].append(
                {"id": data[0], "label": data[1] - 1, "features": list(data[2:])}
            )

    if not output_file:
        output_file = opf_path.split(".")[0] + ".json"

    with open(output_file, "w") as f:
        j.dump(json_data, f)

    logger.info("File converted to %s.", output_file)
