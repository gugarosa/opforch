# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.models import supervised
from opforch.models.semi_supervised import SemiSupervisedOPF
from opforch.stream import splitter
from opforch.utils import constants


@pytest.mark.parametrize("precomputed", [False, True])
def test_supervised_opf_fit_trains_graph(boat_data, precomputed):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)

    assert opf.subgraph.trained is True


def test_supervised_opf_fit_rejects_out_of_range_distance_indices(boat_data):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)

    with pytest.raises(e.ValueError):
        opf.fit(X, Y)


@pytest.mark.parametrize("precomputed", [False, True])
def test_supervised_opf_predict_returns_label_per_sample(boat_data, precomputed):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100
    assert set(preds) <= set(Y.tolist())


def test_supervised_opf_predict_rejects_unfitted_model():
    opf = supervised.SupervisedOPF(device="cpu")

    with pytest.raises(e.BuildError):
        opf.predict(torch.tensor([[0.0]]))


def test_supervised_opf_predict_rejects_untrained_graph(boat_data):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    opf.fit(X, Y)
    opf.subgraph.trained = False

    with pytest.raises(e.BuildError):
        opf.predict(X)


def test_supervised_opf_learn_retains_fitted_classifier(boat_data):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    X_train, X_val, Y_train, Y_val = splitter.split(X, Y, percentage=0.1, random_state=1)

    opf.learn(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert isinstance(opf, supervised.SupervisedOPF)
    assert opf.subgraph.trained is True
    assert len(opf.predict(X_val)) == len(X_val)


def test_supervised_opf_prune_retains_predictable_subset(boat_data):
    X, Y = boat_data
    opf = supervised.SupervisedOPF(device="cpu")
    X_train, X_val, Y_train, Y_val = splitter.split(X, Y, percentage=0.5, random_state=1)

    opf.prune(X_train, Y_train, X_val, Y_val, n_iterations=2)
    preds = opf.predict(X_val)

    assert len(preds) == X_val.shape[0]
    assert 0 < opf.subgraph.n_nodes <= len(X_train)


@pytest.mark.parametrize("model_class", [supervised.SupervisedOPF, SemiSupervisedOPF])
@pytest.mark.parametrize("n_samples", [1, 3])
def test_supervised_opf_fit_supports_single_class(model_class, n_samples):
    features = torch.arange(n_samples, dtype=torch.float32).reshape(-1, 1)
    labels = torch.full((n_samples,), 7)
    opf = model_class(distance="euclidean", device="cpu")

    if model_class is SemiSupervisedOPF:
        opf.fit(features, labels, torch.tensor([[5.0]]))
    else:
        opf.fit(features, labels)

    assert opf.predict(torch.tensor([[0.5], [20.0]])) == [7, 7]
    assert sorted(opf.subgraph.idx_nodes) == list(range(opf.subgraph.n_nodes))
    assert (opf.subgraph.costs < constants.FLOAT_MAX).all()


def test_supervised_opf_predict_marks_winner_and_predecessors():
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")
    opf.fit(torch.tensor([[0.0], [1.0], [9.0], [10.0]]), torch.tensor([0, 0, 1, 1]))

    assert opf.predict(torch.tensor([[0.0]])) == [0]
    assert opf.subgraph.relevant[0] == constants.RELEVANT
    assert opf.subgraph.relevant[1] == constants.RELEVANT
    assert opf.predict(torch.tensor([[10.0]])) == [1]
    assert opf.subgraph.relevant.tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize("validation_indices", [[0], [0, 3]])
def test_supervised_opf_prune_preserves_prototypes_and_predictions(validation_indices):
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    validation = features[validation_indices]
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")

    opf.prune(features, labels, validation, labels[validation_indices], n_iterations=2)

    assert 0 < opf.subgraph.n_nodes < len(features)
    assert set(opf.subgraph.labels.tolist()) == {0, 1}
    assert opf.predict(validation) == labels[validation_indices].tolist()
    torch.testing.assert_close(features, torch.tensor([[0.0], [1.0], [9.0], [10.0]]))


def test_supervised_opf_learn_can_select_zero_accuracy_model():
    features = torch.tensor([[0.0], [10.0]])
    labels = torch.tensor([0, 1])
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")

    opf.learn(features, labels, features, labels.flip(0), n_iterations=1)

    assert opf.subgraph.trained
    assert opf.predict(features) == [0, 1]


def test_supervised_opf_minimax_costs_match_known_forest():
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")
    opf.fit(torch.tensor([[0.0], [1.0], [4.0], [5.0]]), torch.tensor([0, 0, 1, 1]))

    assert opf.subgraph.status.tolist() == [
        constants.STANDARD,
        constants.PROTOTYPE,
        constants.PROTOTYPE,
        constants.STANDARD,
    ]
    torch.testing.assert_close(opf.subgraph.costs, torch.tensor([1.0, 0.0, 0.0, 1.0]).double())
    assert opf.predict(torch.tensor([[-1.0], [0.5], [2.0], [3.0], [6.0]])) == [
        0,
        0,
        0,
        1,
        1,
    ]
