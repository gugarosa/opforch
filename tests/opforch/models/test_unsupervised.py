# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.models import unsupervised
from opforch.subgraphs.knn import KNNSubgraph


def test_unsupervised_opf_min_k_defaults_to_one():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_min_k_accepts_positive_count():
    opf = unsupervised.UnsupervisedOPF(min_k=1, device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_min_k_rejects_zero():
    with pytest.raises(e.ValueError):
        unsupervised.UnsupervisedOPF(min_k=0, device="cpu")


def test_unsupervised_opf_max_k_defaults_to_one():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.max_k == 1


@pytest.mark.parametrize("max_k", [1, 3])
def test_unsupervised_opf_max_k_accepts_positive_count(max_k):
    opf = unsupervised.UnsupervisedOPF(min_k=1, max_k=max_k, device="cpu")

    assert opf.max_k == max_k


def test_unsupervised_opf_max_k_rejects_zero():
    with pytest.raises(e.ValueError):
        unsupervised.UnsupervisedOPF(max_k=0, device="cpu")


def test_unsupervised_opf_max_k_rejects_reversed_range():
    with pytest.raises(e.ValueError):
        unsupervised.UnsupervisedOPF(min_k=2, max_k=1, device="cpu")


@pytest.mark.parametrize("precomputed", [False, True])
def test_unsupervised_opf_fit_trains_graph(boat_data, precomputed):
    X, Y = boat_data
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)

    assert opf.subgraph.trained is True


def test_unsupervised_opf_fit_rejects_out_of_range_distance_indices(boat_data):
    X, Y = boat_data
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)

    with pytest.raises(e.ValueError):
        opf.fit(X, Y)


@pytest.mark.parametrize("precomputed", [False, True])
def test_unsupervised_opf_predict_returns_labels_and_clusters(boat_data, precomputed):
    X, Y = boat_data
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)
    preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100
    assert set(preds) <= set(Y.tolist())
    assert all(0 <= cluster < opf.subgraph.n_clusters for cluster in clusters)


def test_unsupervised_opf_predict_rejects_unfitted_model():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    with pytest.raises(e.BuildError):
        opf.predict(torch.tensor([[0.0]]))


def test_unsupervised_opf_predict_rejects_untrained_graph(boat_data):
    X, Y = boat_data
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    opf.fit(X, Y)
    opf.subgraph.trained = False

    with pytest.raises(e.BuildError):
        opf.predict(X)


def test_unsupervised_opf_propagate_labels_assigns_known_class(boat_data):
    X, Y = boat_data
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    opf.fit(X, Y)

    opf.propagate_labels()

    assert opf.subgraph.pred_labels[0].item() >= 0


def test_unsupervised_opf_propagate_labels_copies_root_labels_in_place():
    labels = torch.tensor([10, 20, 30, 40])
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    opf.subgraph = KNNSubgraph(torch.arange(4.0).reshape(-1, 1), labels, device="cpu")
    opf.subgraph.roots.copy_(torch.tensor([2, 1, 2, 1]))
    opf.subgraph.trained = True
    predictions = opf.subgraph.pred_labels
    predictions.fill_(-1)

    result = opf.propagate_labels()

    assert result is None
    assert opf.subgraph.pred_labels is predictions
    torch.testing.assert_close(predictions, torch.tensor([30, 20, 30, 20]))
    torch.testing.assert_close(opf.subgraph.roots, torch.tensor([2, 1, 2, 1]))
    torch.testing.assert_close(labels, torch.tensor([10, 20, 30, 40]))


@pytest.mark.parametrize("has_graph", [False, True])
def test_unsupervised_opf_propagate_labels_rejects_unfitted_state(has_graph):
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    if has_graph:
        opf.subgraph = KNNSubgraph(torch.tensor([[0.0], [1.0]]), device="cpu")

    with pytest.raises(e.BuildError):
        opf.propagate_labels()


def test_unsupervised_opf_clustering_connects_density_plateau():
    opf = unsupervised.UnsupervisedOPF(device="cpu")
    opf.subgraph = KNNSubgraph(torch.arange(4.0).reshape(-1, 1), device="cpu")
    opf.subgraph.adjacency = torch.tensor([[1], [0], [1], [2]])
    opf.subgraph.densities.fill_(1000)
    opf.subgraph.costs.fill_(999)

    opf._clustering(1)

    assert opf.subgraph.n_clusters == 1
    assert opf.subgraph.cluster_labels.tolist() == [0, 0, 0, 0]


def test_unsupervised_opf_cut_candidates_start_with_unmodified_knn_arcs(monkeypatch):
    features = torch.tensor([[0.0], [0.3], [1.0], [2.0]])
    distances = torch.cdist(features, features)
    graph = KNNSubgraph(features, device="cpu")
    graph.create_arcs_from_matrix(distances, 3)
    neighbours = graph.adjacency.clone()
    opf = unsupervised.UnsupervisedOPF(min_k=1, max_k=3, device="cpu")
    opf.subgraph = graph
    visited = []
    calculate_pdf = graph.calculate_pdf_from_matrix

    def check_candidate(matrix, k):
        torch.testing.assert_close(graph.adjacency, neighbours[:, :k])
        assert not graph.n_plateaus.any()
        visited.append(k)
        calculate_pdf(matrix, k)

    def add_plateaus(k):
        graph.densities.fill_(1000)
        graph.insert_plateaus(k)

    monkeypatch.setattr(graph, "calculate_pdf_from_matrix", check_candidate)
    monkeypatch.setattr(opf, "_clustering", add_plateaus)
    monkeypatch.setattr(opf, "_normalized_cut", lambda k, matrix: 1.0 / k)

    opf._best_minimum_cut(1, 3, distances)

    assert visited == [1, 2, 3, 3]
    assert graph.best_k == 3
