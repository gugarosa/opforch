# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from math import log

import pytest
import torch

import opforch.utils.exception as e
from opforch.models.knn_supervised import KNNSupervisedOPF
from opforch.models.semi_supervised import SemiSupervisedOPF
from opforch.models.supervised import SupervisedOPF
from opforch.models.unsupervised import UnsupervisedOPF
from opforch.subgraphs.knn import KNNSubgraph


@pytest.fixture
def indexed_data(tmp_path):
    X = torch.tensor([[0.0], [10.0], [0.3], [10.8], [2.0], [8.0], [0.1], [9.9]])
    Y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    distances = torch.cdist(X, X)
    path = tmp_path / "distances.pt"
    torch.save(distances, path)
    return X, Y, distances, str(path)


@pytest.mark.parametrize(
    "model_class",
    [SupervisedOPF, KNNSupervisedOPF, SemiSupervisedOPF, UnsupervisedOPF],
)
def test_precomputed_distances_follow_sample_indices(indexed_data, model_class):
    X, Y, distances, path = indexed_data
    train = torch.tensor([3, 0, 1, 2])
    validation = torch.tensor([7, 6])
    unlabeled = torch.tensor([5, 4])
    fitted_indices = train
    direct = model_class(distance="euclidean", device="cpu")
    stored = model_class(distance="euclidean", pre_computed_distance=path, device="cpu")

    for model in (direct, stored):
        if model_class is KNNSupervisedOPF:
            model.fit(X[train], Y[train], X[validation], Y[validation], train, validation)
        elif model_class is SemiSupervisedOPF:
            model.fit(X[train], Y[train], X[unlabeled], train, I_unlabeled=unlabeled)
            fitted_indices = torch.cat((train, unlabeled))
        else:
            model.fit(X[train], Y[train], train)
        if model_class is UnsupervisedOPF:
            model.propagate_labels()

    expected = direct.predict(X[validation])
    actual = stored.predict(X[validation], validation)
    assert actual == expected
    if model_class is UnsupervisedOPF:
        assert len(actual[0]) == len(actual[1]) == len(validation)
    else:
        assert actual == [1, 0]
    torch.testing.assert_close(stored.subgraph.costs, direct.subgraph.costs)
    torch.testing.assert_close(stored.get_distances(), distances[fitted_indices[:, None], fitted_indices])
    empty = stored.predict(X[:0], [])
    assert empty == (([], []) if model_class is UnsupervisedOPF else [])


@pytest.mark.parametrize("model_class", [KNNSupervisedOPF, UnsupervisedOPF])
def test_precomputed_knn_prediction_selects_known_density_conquerors(model_class):
    features = torch.arange(3.0).reshape(-1, 1)
    model = model_class(device="cpu")
    graph = KNNSubgraph(features, torch.tensor([11, 22, 33]), device="cpu")
    graph.best_k = 2
    graph.constant = 1.0
    graph.min_density = 0.0
    graph.max_density = 1.0
    graph.costs.copy_(torch.tensor([300.0, 700.0, 900.0]))
    graph.pred_labels.copy_(graph.labels)
    graph.cluster_labels.copy_(torch.tensor([0, 1, 2]))
    graph.n_clusters = 3
    graph.trained = True
    model.subgraph = graph
    model.pre_computed_distance = True
    model.pre_distances = torch.zeros(6, 6)
    model.pre_distances[:3, 3:] = torch.tensor(
        [[log(2), log(4), log(16)], [log(4), log(8), log(8)], [log(16), log(16), log(4)]]
    )
    model.pre_distances[3:, :3] = model.pre_distances[:3, 3:].T

    actual = model.predict(features, torch.tensor([3, 4, 5]))

    # Query densities are 375.625, 188.3125, and 188.3125 for the two nearest neighbours
    # The first winner is farther away; density-capped ties choose the nearest node in the other queries
    expected_labels = [22, 11, 33]
    if model_class is UnsupervisedOPF:
        assert actual == (expected_labels, [1, 0, 2])
    else:
        assert actual == expected_labels


@pytest.mark.parametrize(
    ("indices", "error"),
    [
        ([-1], e.ValueError),
        ([8], e.ValueError),
        ([0.5], e.TypeError),
        ([[0]], e.SizeError),
        ([0, 1], e.SizeError),
    ],
)
def test_precomputed_prediction_validates_indices(indexed_data, indices, error):
    X, Y, _, path = indexed_data
    model = SupervisedOPF(pre_computed_distance=path, device="cpu")
    model.fit(X, Y)

    with pytest.raises(error):
        model.predict(X[:1], indices)


@pytest.mark.parametrize("method", ["learn", "prune"])
def test_precomputed_learning_and_pruning_preserve_index_alignment(indexed_data, method):
    X, Y, _, path = indexed_data
    train = torch.tensor([3, 0, 1, 2])
    validation = torch.tensor([7, 6])
    model = SupervisedOPF(pre_computed_distance=path, device="cpu")

    getattr(model, method)(
        X[train],
        Y[train],
        X[validation],
        Y[validation],
        n_iterations=2,
        I_train=train,
        I_val=validation,
    )

    assert model.predict(X[validation], validation) == [1, 0]
    torch.testing.assert_close(model.subgraph.features, X[model.subgraph.indices])


def test_semi_supervised_precomputed_defaults_to_consecutive_indices(indexed_data):
    X, Y, _, path = indexed_data
    model = SemiSupervisedOPF(pre_computed_distance=path, device="cpu")

    model.fit(X[:4], Y[:4], X[4:6])

    assert model.predict(X[6:], torch.tensor([6, 7])) == [0, 1]


def test_learning_moves_indices_with_swapped_samples(indexed_data, monkeypatch):
    X, Y, _, path = indexed_data
    train = torch.tensor([0, 2, 1, 3])
    validation = torch.tensor([6, 7])
    model = SupervisedOPF(distance="euclidean", pre_computed_distance=path, device="cpu")
    fit = model.fit
    observed = []

    def record_fit(features, labels, indices):
        torch.testing.assert_close(features, X[indices])
        observed.append(indices.clone())
        fit(features, labels, indices)

    monkeypatch.setattr(model, "fit", record_fit)
    monkeypatch.setattr(
        "opforch.models.supervised.r.generate_uniform_random_number",
        lambda *args: torch.tensor([0.0]),
    )

    model.learn(
        X[train],
        Y[train],
        X[validation],
        torch.tensor([1, 1]),
        n_iterations=2,
        I_train=train,
        I_val=validation,
    )

    assert len(observed) == 2
    assert observed[1][0] == validation[0]
    torch.testing.assert_close(train, torch.tensor([0, 2, 1, 3]))
    torch.testing.assert_close(validation, torch.tensor([6, 7]))
