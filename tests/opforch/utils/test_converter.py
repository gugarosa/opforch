# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import json
import struct

import numpy as np
import pytest

from opforch.utils.converter import opf2csv, opf2json, opf2txt


@pytest.fixture
def binary_opf(tmp_path):
    source = tmp_path / "samples.dat"
    header = struct.pack("<iii", 3, 2, 2)
    records = b"".join(
        struct.pack("<iiff", *row) for row in [(42, 1, 1.25, -2.0), (7, 2, 3.5, 4.25), (99, 1, -6.0, 0.5)]
    )
    source.write_bytes(header + records)
    return source


@pytest.mark.parametrize(("convert", "suffix", "delimiter"), [(opf2txt, ".txt", " "), (opf2csv, ".csv", ",")])
def test_converter_writes_ids_zero_based_labels_and_features(binary_opf, opf_table, convert, suffix, delimiter):
    output = binary_opf.with_name(f"converted{suffix}")

    convert(str(binary_opf), str(output))

    assert output.is_file()
    np.testing.assert_array_equal(np.loadtxt(output, delimiter=delimiter), opf_table.numpy())


def test_opf2json_writes_zero_based_records(binary_opf):
    output = binary_opf.with_name("converted.json")

    opf2json(str(binary_opf), str(output))

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "data": [
            {"id": 42, "label": 0, "features": [1.25, -2.0]},
            {"id": 7, "label": 1, "features": [3.5, 4.25]},
            {"id": 99, "label": 0, "features": [-6.0, 0.5]},
        ]
    }


@pytest.mark.parametrize(("convert", "suffix"), [(opf2txt, ".txt"), (opf2csv, ".csv"), (opf2json, ".json")])
def test_converter_defaults_to_source_path_with_new_suffix(binary_opf, convert, suffix):
    original = binary_opf.read_bytes()

    convert(str(binary_opf))

    assert binary_opf.with_suffix(suffix).is_file()
    assert binary_opf.read_bytes() == original
