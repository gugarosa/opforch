import pytest
import torch

import opforch.utils.exception as e
from opforch.math import distance
from opforch.stream import loader, parser
from opforch.subgraphs import knn

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_knn_subgraph_n_clusters():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.n_clusters == 0


def test_knn_subgraph_n_clusters_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    subgraph.n_clusters = 1

    assert subgraph.n_clusters == 1


def test_knn_subgraph_best_k():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.best_k == 0


def test_knn_subgraph_best_k_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    subgraph.best_k = 1

    assert subgraph.best_k == 1


def test_knn_subgraph_constant():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.constant == 0.0


def test_knn_subgraph_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.density == 0.0


def test_knn_subgraph_density_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    subgraph.density = 2.5

    assert subgraph.density == 2.5


def test_knn_subgraph_min_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.min_density == 0.0


def test_knn_subgraph_max_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.max_density == 0.0


def test_knn_subgraph_calculate_pdf():
    subgraph = knn.KNNSubgraph(X, Y)

    distances = torch.ones((100, 100), dtype=torch.float32)

    subgraph.create_arcs(
        1,
        distance.euclidean_distance,
        pre_computed_distance=True,
        pre_distances=distances,
    )
    subgraph.calculate_pdf(
        1,
        distance.euclidean_distance,
        pre_computed_distance=True,
        pre_distances=distances,
    )

    subgraph.create_arcs(1, distance.euclidean_distance)
    subgraph.calculate_pdf(1, distance.euclidean_distance)

    assert subgraph.min_density != 0
    assert subgraph.max_density != 0


def test_knn_subgraph_create_arcs():
    subgraph = knn.KNNSubgraph(X, Y)

    distances = torch.ones((100, 100), dtype=torch.float32)
    distances.fill_(0.000001)

    subgraph.create_arcs(
        1,
        distance.euclidean_distance,
        pre_computed_distance=True,
        pre_distances=distances,
    )

    max_distances = subgraph.create_arcs(1, distance.euclidean_distance)

    assert len(max_distances) == 1


def test_knn_subgraph_create_arcs_from_matrix():
    subgraph = knn.KNNSubgraph(X, Y)

    dist_matrix = distance.euclidean_distance(subgraph.features, subgraph.features)

    max_distances = subgraph.create_arcs_from_matrix(dist_matrix, 3)

    assert max_distances.shape == (3,)
    assert subgraph.adjacency is not None
    assert subgraph.adjacency.shape == (100, 3)


def test_knn_subgraph_eliminate_maxima_height():
    subgraph = knn.KNNSubgraph(X, Y)

    subgraph.eliminate_maxima_height(2.5)

    assert subgraph.costs[0].item() == 0


def test_knn_subgraph_destroy_arcs():
    subgraph = knn.KNNSubgraph(X, Y)

    dist_matrix = distance.euclidean_distance(subgraph.features, subgraph.features)
    subgraph.create_arcs_from_matrix(dist_matrix, 3)

    subgraph.destroy_arcs()

    assert subgraph.adjacency is None


def test_plateaus_preserve_a_packed_neighbour_prefix():
    subgraph = knn.KNNSubgraph(torch.arange(4.0).reshape(-1, 1), device="cpu")
    subgraph.adjacency = torch.tensor([[1], [0], [1], [2]])
    subgraph.densities.fill_(1000)

    subgraph.insert_plateaus(1)

    assert subgraph.adjacency.tolist() == [[1, -1], [2, 0], [3, 1], [2, -1]]
    assert subgraph.n_plateaus.tolist() == [0, 1, 1, 0]


def test_precomputed_arcs_and_pdf_use_original_indices():
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
def test_knn_rejects_neighbour_counts_that_include_self(k):
    subgraph = knn.KNNSubgraph(torch.arange(3.0).reshape(-1, 1), device="cpu")

    with pytest.raises(e.ValueError, match="k"):
        subgraph.create_arcs(k, distance.euclidean_distance)
