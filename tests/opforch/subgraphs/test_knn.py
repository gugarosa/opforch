import torch

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
