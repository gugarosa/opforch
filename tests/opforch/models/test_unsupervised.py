import torch

from opforch.models import unsupervised
from opforch.stream import loader, parser

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_unsupervised_opf_min_k():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_min_k_setter():
    try:
        opf = unsupervised.UnsupervisedOPF(min_k=0, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(min_k=1, device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_max_k():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.max_k == 1


def test_unsupervised_opf_max_k_setter():
    try:
        opf = unsupervised.UnsupervisedOPF(max_k=0, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(max_k=1, device="cpu")

    assert opf.max_k == 1

    try:
        opf = unsupervised.UnsupervisedOPF(min_k=2, max_k=1, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(min_k=1, max_k=3, device="cpu")

    assert opf.max_k == 3


def test_unsupervised_opf_fit():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    opf.fit(X, Y)

    assert opf.subgraph.trained is True

    opf.pre_computed_distance = True
    try:
        opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)
        opf.fit(X, Y)
    except:
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)
        opf.fit(X, Y)

    assert opf.subgraph.trained is True


def test_unsupervised_opf_predict():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    try:
        opf.fit(X, Y)
        opf.subgraph.trained = False
        _, _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)
    preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100


def test_unsupervised_opf_propagate_labels():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    opf.fit(X, Y)

    opf.propagate_labels()

    assert opf.subgraph.pred_labels[0].item() >= 0


def test_clustering_connects_a_density_plateau():
    from opforch.subgraphs import KNNSubgraph

    opf = unsupervised.UnsupervisedOPF(device="cpu")
    opf.subgraph = KNNSubgraph(torch.arange(4.0).reshape(-1, 1), device="cpu")
    opf.subgraph.adjacency = torch.tensor([[1], [0], [1], [2]])
    opf.subgraph.densities.fill_(1000)
    opf.subgraph.costs.fill_(999)

    opf._clustering(1)

    assert opf.subgraph.n_clusters == 1
    assert opf.subgraph.cluster_labels.tolist() == [0, 0, 0, 0]


def test_each_cut_candidate_starts_with_unmodified_knn_arcs(monkeypatch):
    from opforch.subgraphs import KNNSubgraph

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
