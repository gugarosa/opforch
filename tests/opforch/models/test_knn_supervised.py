# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.models.knn_supervised import KNNSupervisedOPF


def test_knn_supervised_opf_max_k_defaults_to_one():
    opf = KNNSupervisedOPF(device="cpu")

    assert opf.max_k == 1


def test_knn_supervised_opf_max_k_accepts_positive_count():
    opf = KNNSupervisedOPF(max_k=3, device="cpu")

    assert opf.max_k == 3


def test_knn_supervised_opf_max_k_rejects_zero():
    with pytest.raises(e.ValueError):
        KNNSupervisedOPF(max_k=0, device="cpu")


@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_supervised_opf_fit_trains_graph(boat_data, precomputed):
    X, Y = boat_data
    opf = KNNSupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y, X, Y)

    assert opf.subgraph.trained is True


def test_knn_supervised_opf_fit_rejects_out_of_range_distance_indices(boat_data):
    X, Y = boat_data
    opf = KNNSupervisedOPF(device="cpu")
    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)

    with pytest.raises(e.ValueError):
        opf.fit(X, Y, X, Y)


@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_supervised_opf_predict_returns_label_per_sample(boat_data, precomputed):
    X, Y = boat_data
    opf = KNNSupervisedOPF(device="cpu")
    if precomputed:
        opf.pre_computed_distance = True
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y, X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100
    assert set(preds) <= set(Y.tolist())


def test_knn_supervised_opf_predict_rejects_unfitted_model():
    opf = KNNSupervisedOPF(device="cpu")

    with pytest.raises(e.BuildError):
        opf.predict(torch.tensor([[0.0]]))


@pytest.mark.parametrize("previously_fitted", [False, True])
def test_knn_supervised_opf_predict_rejects_failed_fit(previously_fitted):
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    validation = torch.tensor([[0.1], [9.9]])
    opf = KNNSupervisedOPF(distance="euclidean", device="cpu")
    if previously_fitted:
        opf.fit(features, labels, validation, torch.tensor([0, 1]))

    with pytest.raises(e.SizeError):
        opf.fit(features, labels, validation, torch.tensor([0]))

    assert opf.subgraph.trained is False
    with pytest.raises(e.BuildError):
        opf.predict(validation)


def test_knn_supervised_opf_fit_reuses_training_distances(monkeypatch):
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    validation = torch.tensor([[0.1], [9.9]])
    opf = KNNSupervisedOPF(max_k=3, distance="euclidean", device="cpu")
    distance_fn = opf.distance_fn
    observed = []

    def record_distances(left, right):
        observed.append((tuple(left.shape), tuple(right.shape)))
        return distance_fn(left, right)

    monkeypatch.setattr(opf, "distance_fn", record_distances)

    opf.fit(features, labels, validation, torch.tensor([0, 1]))

    assert opf.subgraph.trained is True
    assert opf.subgraph.best_k == 1
    assert observed.count(((4, 1), (4, 1))) == 1
    assert ((4, 1), (2, 1)) in observed
    assert opf.predict(validation) == [0, 1]
