"""General-purpose mathematical methods (tensor-based)."""

from typing import List, Tuple, Union

import torch

import opforch.math.distance as d
import opforch.utils.exception as e
from opforch.utils import logging

logger = logging.get_logger(__name__)


def _label_tensors(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor(labels)
    preds = torch.as_tensor(preds, device=labels.device)
    if (
        labels.ndim != 1
        or preds.ndim != 1
        or labels.shape != preds.shape
        or not len(labels)
    ):
        raise e.SizeError(
            "Labels and predictions must be non-empty vectors of equal length"
        )
    for values in (labels, preds):
        if values.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise e.TypeError("Labels and predictions must be integers")
        if (values < 0).any():
            raise e.ValueError("Labels and predictions must be non-negative")
    return labels.long(), preds.long()


def _confusion_counts(
    labels: torch.Tensor, preds: torch.Tensor, n_labels: int, n_preds: int
) -> torch.Tensor:
    return torch.bincount(
        labels * n_preds + preds, minlength=n_labels * n_preds
    ).reshape(n_labels, n_preds)


def confusion_matrix(
    labels: Union[torch.Tensor, List[int]],
    preds: Union[torch.Tensor, List[int]],
) -> torch.Tensor:
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels: True labels.
        preds: Predicted labels.

    Returns:
        The confusion matrix of shape (C, C), where C includes classes present
        in either labels or predictions. Class IDs must be non-negative integers.

    """

    labels, preds = _label_tensors(labels, preds)
    n_class = int(torch.maximum(labels.max(), preds.max()).item()) + 1
    return _confusion_counts(labels, preds, n_class, n_class).float()


def normalize(array: torch.Tensor) -> torch.Tensor:
    """Normalizes an input tensor (z-score normalization).

    Args:
        array: Tensor to be normalized.

    Returns:
        Normalized tensor with zero mean and unit standard deviation per
        nonconstant column. Constant columns become zeros.

    """

    mean = array.mean(dim=0)
    std = array.std(dim=0, correction=0)
    std = torch.where(std == 0, 1, std)

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
        The OPF accuracy measure between 0 and 1. The class range covers both
        labels and predictions; undefined error rates for absent classes are
        omitted, as for other zero denominators.

    """

    labels, preds = _label_tensors(labels, preds)
    n_class = int(torch.maximum(labels.max(), preds.max()).item()) + 1

    errors = torch.zeros(n_class, 2, dtype=torch.float64, device=labels.device)
    counts = torch.bincount(labels, minlength=n_class).double()

    wrong = labels != preds
    errors[:, 0] = torch.bincount(preds[wrong], minlength=n_class)
    errors[:, 1] = torch.bincount(labels[wrong], minlength=n_class)

    errors[:, 1] /= counts
    errors[:, 0] /= counts.sum() - counts
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

    labels, preds = _label_tensors(labels, preds)

    n_class = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=n_class).double()

    wrong = labels != preds
    errors = torch.bincount(labels[wrong], minlength=n_class).double()

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

    labels, preds = _label_tensors(labels, preds)
    c_matrix = _confusion_counts(
        labels, preds, int(labels.max().item()) + 1, int(preds.max().item()) + 1
    )
    _purity = c_matrix.max(dim=0).values.sum() / len(labels)

    return _purity.item()
