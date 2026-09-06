# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from opforch.core.subgraph import Subgraph
from opforch.models.semi_supervised import SemiSupervisedOPF
from opforch.utils import constants


def test_semi_supervised_opf_fit_trains_combined_graph(boat_data):
    X, Y = boat_data
    original_features = X.clone()
    original_labels = Y.clone()
    opf = SemiSupervisedOPF(device="cpu")

    opf.fit(X, Y, X)

    assert opf.subgraph.trained is True
    assert opf.subgraph.n_nodes == 200
    torch.testing.assert_close(X, original_features)
    torch.testing.assert_close(Y, original_labels)


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("custom_indices", [False, True])
def test_semi_supervised_opf_fit_preserves_labeled_mst_and_canonical_state(
    tmp_path, monkeypatch, precomputed, custom_indices
):
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0], [0.25], [9.75], [0.1], [9.9]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    indices = torch.tensor([3, 0, 7, 1, 6, 2, 4, 5], dtype=torch.int32) if custom_indices else torch.arange(8)
    originals = [tensor.clone() for tensor in (features, labels, indices)]

    indexed_features = torch.empty_like(features)
    indexed_features[indices.long()] = features
    path = tmp_path / "distances.pt"
    if precomputed:
        torch.save(torch.cdist(indexed_features.float(), indexed_features.float()), path)

    opf = SemiSupervisedOPF(
        distance="euclidean", pre_computed_distance=str(path) if precomputed else None, device="cpu"
    )
    canonical = Subgraph(features[:6], torch.tensor([0, 0, 1, 1, 0, 0]), indices[:6], device="cpu")
    # The labeled 0-1-9-10 chain crosses its only class boundary between the middle samples
    canonical.status[:4] = torch.tensor(
        [constants.STANDARD, constants.PROTOTYPE, constants.PROTOTYPE, constants.STANDARD], dtype=torch.int8
    )
    canonical.preds[:4] = torch.tensor([constants.NIL, 0, 1, 2])
    observed = []
    compete = opf._compete

    def check_combined_state(distances):
        graph = opf.subgraph
        observed.append(graph.n_nodes)
        assert graph.n_nodes == 6
        assert graph.n_features == 1
        assert graph.trained is False
        assert graph.adjacency is None
        assert graph.idx_nodes == []
        for name, value in vars(canonical).items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(getattr(graph, name), value)

        compete(distances)

    monkeypatch.setattr(opf, "_compete", check_combined_state)

    opf.fit(
        features[:4],
        labels,
        features[4:6],
        I_train=indices[:4] if custom_indices else None,
        I_unlabeled=indices[4:6] if custom_indices else None,
    )

    graph = opf.subgraph
    assert observed == [6]
    assert graph.trained is True
    assert graph.n_nodes == 6
    assert graph.n_features == 1
    assert sorted(graph.idx_nodes) == list(range(6))
    for name, value in vars(canonical).items():
        if isinstance(value, torch.Tensor):
            actual = getattr(graph, name)
            assert actual.shape == value.shape
            assert actual.dtype == value.dtype
            assert actual.device == torch.device("cpu")
    torch.testing.assert_close(graph.indices, indices[:6].long())
    torch.testing.assert_close(graph.status, canonical.status)
    torch.testing.assert_close(graph.labels, torch.tensor([0, 0, 1, 1, 0, 1]))
    torch.testing.assert_close(graph.pred_labels, torch.tensor([0, 0, 1, 1, 0, 1]))
    torch.testing.assert_close(graph.costs, torch.tensor([0.75, 0.0, 0.0, 0.75, 0.75, 0.75], dtype=torch.float64))
    torch.testing.assert_close(opf.get_distances(), torch.cdist(features[:6].float(), features[:6].float()))

    assert opf.predict(features[6:], indices[6:]) == [0, 1]
    for actual, original in zip((features, labels, indices), originals):
        torch.testing.assert_close(actual, original)
