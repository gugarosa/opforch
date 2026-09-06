# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.math import general
from opforch.stream import splitter


def test_confusion_matrix_counts_correct_predictions():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    c_matrix = general.confusion_matrix(labels, preds)

    assert c_matrix.shape == (2, 2)
    torch.testing.assert_close(c_matrix, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_normalize_maps_single_sample_to_zero():
    array = torch.tensor([[1.0, 1.0, 1.0, 2.0]])

    norm_array = general.normalize(array)

    assert norm_array.shape == array.shape
    torch.testing.assert_close(norm_array, torch.zeros_like(array))


def test_opf_accuracy_penalizes_missing_class():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 0, 0]

    acc = general.opf_accuracy(labels, preds)

    assert acc == 0.5


def test_opf_accuracy_per_label_reports_class_scores():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 0, 0]

    acc_per_label = general.opf_accuracy_per_label(labels, preds)

    assert acc_per_label.shape == (2,)
    torch.testing.assert_close(acc_per_label, torch.tensor([1.0, 0.0]).double())


def test_pre_compute_distance_saves_pairwise_tensor(tmp_path, boat_data):
    X, Y = boat_data
    X_train, _, _, _ = splitter.split(X, Y, 0.5, 1)
    output = tmp_path / "distances.pt"

    general.pre_compute_distance(X_train, str(output), "log_squared_euclidean")

    assert output.is_file()
    distances = torch.load(output, weights_only=True)
    assert distances.shape == (50, 50)
    assert distances.device == torch.device("cpu")
    assert torch.isfinite(distances).all()


def test_pre_compute_distance_preserves_known_euclidean_matrix(tmp_path):
    features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-3.0, -4.0]])
    output = tmp_path / "distances.pt"

    general.pre_compute_distance(features, str(output), "euclidean")

    distances = torch.load(output, weights_only=True)
    torch.testing.assert_close(distances, torch.tensor([[0.0, 5.0, 5.0], [5.0, 0.0, 10.0], [5.0, 10.0, 0.0]]))


def test_purity_scores_perfect_clusters_as_one():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    purity = general.purity(labels, preds)

    assert purity == 1


def test_confusion_matrix_includes_predicted_only_classes():
    actual = general.confusion_matrix([0, 0, 1, 1], [2, 0, 1, 1])

    torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))


def test_opf_accuracy_includes_predicted_only_classes():
    # The extra class adds a 1/4 false-positive rate to the 1/2 false-negative rate
    assert general.opf_accuracy([0, 0, 1, 1], [2, 0, 1, 1]) == pytest.approx(0.875)


def test_purity_allows_more_clusters_than_classes():
    assert general.purity([0, 0, 1, 1], [0, 1, 2, 3]) == 1.0


def test_normalize_preserves_nonconstant_columns():
    actual = general.normalize(torch.tensor([[1.0, 2.0], [1.0, 4.0]]))

    torch.testing.assert_close(actual, torch.tensor([[0.0, -1.0], [0.0, 1.0]]))


@pytest.mark.parametrize(
    ("labels", "predictions", "error"),
    [
        ([0, 1], [0], e.SizeError),
        ([], [], e.SizeError),
        ([0, -1], [0, 1], e.ValueError),
        ([0, 1], [0, 0.5], e.TypeError),
    ],
)
@pytest.mark.parametrize(
    "metric",
    [general.confusion_matrix, general.opf_accuracy, general.opf_accuracy_per_label],
)
def test_label_metrics_reject_invalid_inputs(metric, labels, predictions, error):
    with pytest.raises(error):
        metric(labels, predictions)
