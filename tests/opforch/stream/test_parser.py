# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from opforch.stream.loader import load_csv
from opforch.stream.parser import parse_loader


@pytest.mark.parametrize("as_list", [False, True])
def test_parse_loader_extracts_features_and_zero_based_labels(opf_table, as_list):
    original = opf_table.clone()
    data = opf_table.tolist() if as_list else opf_table

    features, labels = parse_loader(data)

    torch.testing.assert_close(features, torch.tensor([[1.25, -2.0], [3.5, 4.25], [-6.0, 0.5]]))
    torch.testing.assert_close(labels, torch.tensor([0, 1, 0]))
    torch.testing.assert_close(opf_table, original)


def test_parse_loader_accepts_single_zero_based_label():
    features, labels = parse_loader([[7, 0, 1.5], [9, 0, 2.5]])

    torch.testing.assert_close(features, torch.tensor([[1.5], [2.5]]))
    torch.testing.assert_close(labels, torch.tensor([0, 0]))


def test_parse_loader_accepts_tables_without_feature_columns():
    table = torch.tensor([[4, 0], [7, 1]], dtype=torch.int32)

    features, labels = parse_loader(table)

    torch.testing.assert_close(features, torch.empty(2, 0))
    torch.testing.assert_close(labels, torch.tensor([0, 1]))


def test_parse_loader_casts_fractional_labels_before_validation():
    table = torch.tensor([[4, 0.75, 1.5], [7, 1.75, 2.5]], dtype=torch.float64)

    features, labels = parse_loader(table)

    torch.testing.assert_close(features, torch.tensor([[1.5], [2.5]]))
    torch.testing.assert_close(labels, torch.tensor([0, 1]))


def test_parse_loader_reads_boat_features_and_labels(data_dir):
    table = load_csv(str(data_dir / "boat.csv"))

    features, labels = parse_loader(table)

    assert features.shape == (100, 2)
    assert labels.shape == (100,)
    torch.testing.assert_close(features, table[:, 2:].float())
    torch.testing.assert_close(labels, table[:, 1].long())


@pytest.mark.parametrize(
    "data",
    [
        None,
        torch.tensor([]),
        torch.empty(0, 4),
        torch.ones(4),
        torch.ones(2, 2, 2),
        torch.tensor([[4], [7]]),
        torch.ones(4, 4),
        torch.tensor([[0, 0, 1.0], [1, 3, 2.0]]),
        torch.tensor([[0, -1, 1.0], [1, 0, 2.0]]),
    ],
)
def test_parse_loader_returns_none_for_malformed_data(data):
    assert parse_loader(data) == (None, None)


@pytest.mark.parametrize("error_class", [RuntimeError, torch.OutOfMemoryError])
def test_parse_loader_propagates_unexpected_backend_errors(opf_table, monkeypatch, error_class):
    error = error_class("backend failure")

    def fail_unique(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "unique", fail_unique)

    with pytest.raises(error_class) as caught:
        parse_loader(opf_table)

    assert caught.value is error
