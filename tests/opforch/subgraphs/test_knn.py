# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.math import distance
from opforch.subgraphs import knn


def test_knn_subgraph_n_clusters_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.n_clusters == 0


def test_knn_subgraph_n_clusters_accepts_updated_count(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    subgraph.n_clusters = 1

    assert subgraph.n_clusters == 1


def test_knn_subgraph_best_k_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.best_k == 0


def test_knn_subgraph_best_k_accepts_updated_count(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    subgraph.best_k = 1

    assert subgraph.best_k == 1


def test_knn_subgraph_constant_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.constant == 0.0


def test_knn_subgraph_density_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.density == 0.0


def test_knn_subgraph_density_accepts_updated_scale(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    subgraph.density = 2.5

    assert subgraph.density == 2.5


def test_knn_subgraph_min_density_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.min_density == 0.0


def test_knn_subgraph_max_density_defaults_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    assert subgraph.max_density == 0.0


@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_subgraph_calculate_pdf_assigns_positive_densities(boat_data, precomputed):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")
    distances = torch.ones((100, 100), dtype=torch.float32)

    subgraph.create_arcs(1, distance.euclidean_distance, precomputed, distances)
    subgraph.calculate_pdf(1, distance.euclidean_distance, precomputed, distances)

    assert subgraph.min_density != 0
    assert subgraph.max_density != 0
    assert (subgraph.densities > 0).all()


@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_subgraph_create_arcs_returns_maximum_neighbor_distances(boat_data, precomputed):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")
    distances = torch.full((100, 100), 0.000001)

    max_distances = subgraph.create_arcs(1, distance.euclidean_distance, precomputed, distances)

    assert len(max_distances) == 1
    assert subgraph.adjacency.shape == (100, 1)
    assert (subgraph.adjacency[:, 0] != torch.arange(100)).all()


def test_knn_subgraph_create_arcs_from_matrix_builds_requested_neighbors(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")
    dist_matrix = distance.euclidean_distance(subgraph.features, subgraph.features)

    max_distances = subgraph.create_arcs_from_matrix(dist_matrix, 3)

    assert max_distances.shape == (3,)
    assert subgraph.adjacency is not None
    assert subgraph.adjacency.shape == (100, 3)


def test_knn_subgraph_eliminate_maxima_height_clamps_costs_to_zero(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")

    subgraph.eliminate_maxima_height(2.5)

    assert subgraph.costs[0].item() == 0


def test_knn_subgraph_destroy_arcs_clears_adjacency_and_plateaus(boat_data):
    subgraph = knn.KNNSubgraph(*boat_data, device="cpu")
    dist_matrix = distance.euclidean_distance(subgraph.features, subgraph.features)
    subgraph.create_arcs_from_matrix(dist_matrix, 3)
    plateaus = subgraph.n_plateaus
    plateaus.fill_(1)

    subgraph.destroy_arcs()

    assert subgraph.adjacency is None
    assert subgraph.n_plateaus is plateaus
    assert not plateaus.any()


def test_knn_subgraph_plateaus_preserve_packed_neighbour_prefix():
    subgraph = knn.KNNSubgraph(torch.arange(4.0).reshape(-1, 1), device="cpu")
    subgraph.adjacency = torch.tensor([[1], [0], [1], [2]])
    subgraph.densities.fill_(1000)

    subgraph.insert_plateaus(1)

    assert subgraph.adjacency.tolist() == [[1, -1], [2, 0], [3, 1], [2, -1]]
    assert subgraph.n_plateaus.tolist() == [0, 1, 1, 0]


def test_knn_subgraph_precomputed_arcs_and_pdf_use_original_indices():
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0]])
    indices = torch.tensor([3, 0, 2])
    distances = torch.cdist(features, features)
    direct = knn.KNNSubgraph(features[indices], device="cpu")
    stored = knn.KNNSubgraph(features[indices], I=indices, device="cpu")

    direct.create_arcs(1, distance.euclidean_distance)
    direct.calculate_pdf(1, distance.euclidean_distance)
    stored.create_arcs(1, distance.euclidean_distance, True, distances)
    stored.calculate_pdf(1, distance.euclidean_distance, True, distances)

    torch.testing.assert_close(stored.adjacency, direct.adjacency)
    torch.testing.assert_close(stored.radii, direct.radii)
    torch.testing.assert_close(stored.densities, direct.densities)


@pytest.mark.parametrize("k", [0, 3, 4])
def test_knn_subgraph_rejects_neighbour_counts_that_include_self(k):
    subgraph = knn.KNNSubgraph(torch.arange(3.0).reshape(-1, 1), device="cpu")

    with pytest.raises(e.ValueError, match="k"):
        subgraph.create_arcs(k, distance.euclidean_distance)
