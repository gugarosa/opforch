"""General-purpose mathematical methods (tensor-based)."""

from typing import List, Union

import torch

import opforch.math.distance as d
from opforch.utils import logging

logger = logging.get_logger(__name__)


def confusion_matrix(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> torch.Tensor:
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels: True labels.
        preds: Predicted labels.

    Returns:
        The confusion matrix of shape (C, C).

    """

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.int64)

    n_class = int(labels.max().item()) + 1
    indices = labels * n_class + preds

    c_matrix = torch.zeros(n_class * n_class, dtype=torch.float32, device=labels.device)
    c_matrix.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=torch.float32))

    return c_matrix.reshape(n_class, n_class)


def normalize(array: torch.Tensor) -> torch.Tensor:
    """Normalizes an input tensor (z-score normalization).

    Args:
        array: Tensor to be normalized.

    Returns:
        Normalized tensor with zero mean and unit standard deviation per column.

    """

    mean = array.mean(dim=0)
    std = array.std(dim=0, correction=0)

    return (array - mean) / std


def opf_accuracy(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> float:
    """Calculates accuracy using the OPF-style measure.

    Args:
        labels: True labels.
        preds: Predicted labels.

    Returns:
        The OPF accuracy measure between 0 and 1.

    """

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.int64)

    n_class = int(labels.max().item()) + 1

    errors = torch.zeros(n_class, 2, dtype=torch.float64)
    counts = torch.bincount(labels, minlength=n_class).double()

    wrong = labels != preds
    for i in range(len(labels)):
        if wrong[i]:
            errors[preds[i], 0] += 1
            errors[labels[i], 1] += 1

    errors[:, 1] /= counts
    errors[:, 0] /= (counts.sum() - counts)
    errors = torch.nansum(errors, dim=1)

    accuracy = 1 - (errors.sum() / (2 * n_class))

    return accuracy.item()


def opf_accuracy_per_label(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> torch.Tensor:
    """Calculates per-label accuracy using the OPF-style measure.

    Args:
        labels: True labels.
        preds: Predicted labels.

    Returns:
        Tensor of per-class accuracy values between 0 and 1.

    """

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.int64)

    n_class = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=n_class).double()

    errors = torch.zeros(n_class, dtype=torch.float64)
    wrong = labels != preds
    for i in range(len(labels)):
        if wrong[i]:
            errors[labels[i]] += 1

    errors /= counts
    accuracy = 1 - errors

    return accuracy


def pre_compute_distance(
    data: torch.Tensor,
    output: str,
    distance: str = "log_squared_euclidean",
) -> None:
    """Pre-computes a distance matrix and saves it.

    Args:
        data: Tensor of samples (N, D).
        output: File path to save the distance matrix.
        distance: Distance metric name.

    """

    logger.info("Pre-computing distances ...")

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    dist_fn = d.DISTANCES[distance]
    distances = dist_fn(data, data)

    torch.save(distances.cpu(), output)

    logger.info("Distances saved to: %s.", output)


def purity(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> float:
    """Calculates the purity measure of an unsupervised technique.

    Args:
        labels: True labels.
        preds: Assigned cluster labels.

    Returns:
        The purity measure.

    """

    c_matrix = confusion_matrix(labels, preds)
    _purity = c_matrix.max(dim=0).values.sum() / len(labels)

    return _purity.item()
