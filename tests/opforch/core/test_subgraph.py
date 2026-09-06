# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.core import subgraph
from opforch.utils import constants


def test_subgraph_n_nodes_defaults_to_zero():
    s = subgraph.Subgraph()

    assert s.n_nodes == 0


def test_subgraph_n_features_defaults_to_zero():
    s = subgraph.Subgraph()

    assert s.n_features == 0


def test_subgraph_trained_defaults_to_false():
    s = subgraph.Subgraph()

    assert s.trained is False


@pytest.mark.parametrize("suffix", [".csv", ".json", ".txt"])
def test_subgraph_load_reads_supported_table_formats(data_dir, suffix):
    s = subgraph.Subgraph(device="cpu")

    X, Y = s._load(str(data_dir / f"boat{suffix}"))

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_subgraph_load_rejects_unsupported_extension(tmp_path):
    s = subgraph.Subgraph(device="cpu")

    with pytest.raises(e.ArgumentError):
        s._load(str(tmp_path / "boat"))


def test_subgraph_build_initializes_default_indices(boat_data):
    s = subgraph.Subgraph(device="cpu")
    X, Y = boat_data

    s._build(X, Y, None)

    assert s.n_nodes == 100
    assert s.n_features == 2
    torch.testing.assert_close(s.indices, torch.arange(100))


def test_subgraph_build_preserves_supplied_indices(boat_data):
    s = subgraph.Subgraph(device="cpu")
    X, Y = boat_data
    I = Y

    s._build(X, Y, I)

    assert s.n_nodes == 100
    assert s.n_features == 2
    torch.testing.assert_close(s.indices, I)


def test_subgraph_from_file_initializes_dimensions(data_dir):
    s = subgraph.Subgraph(from_file=str(data_dir / "boat.txt"), device="cpu")

    assert s.n_nodes == 100
    assert s.n_features == 2


def test_subgraph_from_tensors_initializes_dimensions():
    X = torch.randn(10, 3)
    Y = torch.zeros(10, dtype=torch.int64)

    s = subgraph.Subgraph(X, Y)

    assert s.n_nodes == 10
    assert s.n_features == 3


def test_subgraph_destroy_arcs_clears_adjacency_and_plateaus(boat_data):
    s = subgraph.Subgraph(*boat_data, device="cpu")
    s.adjacency = torch.zeros(100, 1, dtype=torch.int64)
    s.n_plateaus.fill_(1)

    s.destroy_arcs()

    assert s.adjacency is None
    assert not s.n_plateaus.any()


def test_subgraph_mark_nodes_marks_relevance(boat_data):
    s = subgraph.Subgraph(*boat_data, device="cpu")

    s.mark_nodes(0)

    assert s.relevant[0].item() == constants.RELEVANT


def test_subgraph_reset_clears_predecessors_and_relevance(boat_data):
    s = subgraph.Subgraph(*boat_data, device="cpu")
    s.preds.fill_(0)
    s.relevant.fill_(constants.RELEVANT)

    s.reset()

    assert (s.preds == constants.NIL).all()
    assert (s.relevant == constants.IRRELEVANT).all()


def test_subgraph_to_keeps_device_metadata_coherent(boat_data):
    s = subgraph.Subgraph(*boat_data, device="cpu")

    result = s.to("cpu")

    assert result is s
    assert s.device == torch.device("cpu")
    assert s.features.device == s.device


def test_subgraph_to_preserves_empty_index_tensor():
    s = subgraph.Subgraph(device="cpu").to("cpu")

    assert s.indices.shape == (0,)
    assert s.indices.dtype == torch.int64


def test_subgraph_mark_nodes_follows_complete_predecessor_chain():
    s = subgraph.Subgraph(torch.zeros(4, 1), device="cpu")
    s.preds = torch.tensor([constants.NIL, 0, 1, 2])

    s.mark_nodes(3)
    s.mark_nodes(3)

    assert s.relevant.tolist() == [constants.RELEVANT] * 4


def test_subgraph_mark_nodes_rejects_missing_node():
    s = subgraph.Subgraph(torch.zeros(4, 1), device="cpu")

    with pytest.raises(e.ValueError):
        s.mark_nodes(constants.NIL)


@pytest.mark.parametrize(
    ("features", "labels", "indices", "error"),
    [
        (torch.zeros(2), None, None, e.SizeError),
        (torch.zeros(2, 1), torch.zeros(1), None, e.SizeError),
        (torch.zeros(2, 1), None, [0], e.SizeError),
        (torch.zeros(2, 1), None, [-1, 0], e.ValueError),
        (torch.zeros(2, 1), None, [0, 0.5], e.TypeError),
    ],
)
def test_subgraph_validates_sample_alignment(features, labels, indices, error):
    with pytest.raises(error):
        subgraph.Subgraph(features, labels, indices, device="cpu")
