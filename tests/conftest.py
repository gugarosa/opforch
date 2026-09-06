# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

import pytest
import torch

from opforch.stream.loader import load_csv
from opforch.stream.parser import parse_loader


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def boat_data(data_dir):
    return parse_loader(load_csv(str(data_dir / "boat.csv"), device="cpu"))


@pytest.fixture
def opf_table():
    return torch.tensor(
        [[42, 0, 1.25, -2.0], [7, 1, 3.5, 4.25], [99, 0, -6.0, 0.5]],
        dtype=torch.float64,
    )
