# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import json

import pytest
import torch

from opforch.stream.loader import load_csv, load_json, load_txt


@pytest.mark.parametrize(
    ("load", "suffix", "content"),
    [
        (load_csv, ".csv", "42,0,1.25,-2\n7,1,3.5,4.25\n99,0,-6,0.5\n"),
        (load_txt, ".txt", "42 0 1.25 -2\n7 1 3.5 4.25\n99 0 -6 0.5\n"),
        (
            load_json,
            ".json",
            json.dumps(
                {
                    "data": [
                        {"id": 42, "label": 0, "features": [1.25, -2.0]},
                        {"id": 7, "label": 1, "features": [3.5, 4.25]},
                        {"id": 99, "label": 0, "features": [-6.0, 0.5]},
                    ]
                }
            ),
        ),
    ],
)
def test_loader_preserves_opf_table_contents(tmp_path, opf_table, load, suffix, content):
    source = tmp_path / f"samples{suffix}"
    source.write_text(content, encoding="utf-8")

    actual = load(str(source), device="cpu")

    torch.testing.assert_close(actual, opf_table)
    assert actual.device == torch.device("cpu")


@pytest.mark.parametrize(("load", "suffix"), [(load_csv, ".csv"), (load_txt, ".txt"), (load_json, ".json")])
def test_loader_reads_boat_table_shape(data_dir, load, suffix):
    actual = load(str(data_dir / f"boat{suffix}"))

    assert actual.shape == (100, 4)
    assert actual.dtype == torch.float64


@pytest.mark.parametrize(("load", "suffix"), [(load_csv, ".csv"), (load_txt, ".txt"), (load_json, ".json")])
def test_loader_returns_none_for_missing_file(tmp_path, load, suffix):
    assert load(str(tmp_path / f"missing{suffix}")) is None
