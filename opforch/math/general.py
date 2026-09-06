# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Tensor-based normalization, label metrics, and pairwise score persistence.

Label metrics accept nonempty, equally sized vectors of nonnegative IDs with uint8, int8, int16, int32, or int64 dtypes.
Predictions are moved to the device inferred for the true labels, and inputs are not modified.
Class axes retain gaps in label IDs rather than remapping IDs to consecutive values.

"""

from __future__ import annotations

import torch

import opforch.math.distance as d
import opforch.utils.exception as e
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


def _label_tensors(
    labels: torch.Tensor | list[int],
    preds: torch.Tensor | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor(labels)
    preds = torch.as_tensor(preds, device=labels.device)

    if labels.ndim != 1 or preds.ndim != 1 or labels.shape != preds.shape or not len(labels):
        raise e.SizeError("`labels` and `preds` must be nonempty vectors of equal length.")

    for values in (labels, preds):
        if values.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise e.TypeError("`labels` and `preds` must have uint8, int8, int16, int32, or int64 dtypes.")
        if (values < 0).any():
            raise e.ValueError("`labels` and `preds` must be nonnegative.")

    return labels.long(), preds.long()


def _confusion_counts(labels: torch.Tensor, preds: torch.Tensor, n_labels: int, n_preds: int) -> torch.Tensor:
    return torch.bincount(labels * n_preds + preds, minlength=n_labels * n_preds).reshape(n_labels, n_preds)


def confusion_matrix(
    labels: torch.Tensor | list[int],
    preds: torch.Tensor | list[int],
) -> torch.Tensor:
    """Count true and predicted label pairs in a square confusion matrix.

    The class count C is one plus the largest ID in either input, including predicted-only classes.
    Missing IDs retain zero-filled rows and columns.

    Args:
        labels: Nonempty (N,) integer tensor or list of nonnegative true-label IDs.
        preds: (N,) integer tensor or list of nonnegative predicted IDs, moved to the labels' device.

    Returns:
        torch.Tensor: Float32 (C, C) counts on the labels' device, indexed by true rows and predicted columns.

    Raises:
        opforch.utils.exception.SizeError: If either vector is empty, not one-dimensional, or mismatched in length.
        opforch.utils.exception.TypeError: If either vector is not uint8 or a signed integer dtype.
        opforch.utils.exception.ValueError: If either vector contains a negative label.

    """

    labels, preds = _label_tensors(labels, preds)
    n_class = int(torch.maximum(labels.max(), preds.max()).item()) + 1
    return _confusion_counts(labels, preds, n_class, n_class).float()


def normalize(array: torch.Tensor) -> torch.Tensor:
    """Apply population z-score normalization along dimension zero.

    The standard deviation uses correction=0, and zero standard deviations are replaced by one.
    Constant columns therefore become zero, while nonconstant columns use their population standard deviation.
    The input is not modified.

    Args:
        array: Floating-point or complex tensor with samples along dimension zero, usually shaped (N, D).

    Returns:
        torch.Tensor: Normalized values with the input shape, dtype, and device, with constant columns mapped to zero.

    """

    mean = array.mean(dim=0)
    std = array.std(dim=0, correction=0)
    std = torch.where(std == 0, 1, std)

    return (array - mean) / std


def opf_accuracy(
    labels: torch.Tensor | list[int],
    preds: torch.Tensor | list[int],
) -> float:
    """Compute the OPF accuracy measure from per-class error rates.

    This is not fraction-correct accuracy: it is one minus the sum of per-class false-positive and
    false-negative rates divided by 2 * C. C is one plus the largest ID in either input, including gaps.
    Undefined rates from zero denominators are omitted with nansum, without reducing C.

    Args:
        labels: Nonempty (N,) integer tensor or list of nonnegative true-label IDs.
        preds: (N,) integer tensor or list of nonnegative predicted IDs, moved to the labels' device.

    Returns:
        float: OPF score in [0, 1], using both false-positive and false-negative rates.

    Raises:
        opforch.utils.exception.SizeError: If either vector is empty, not one-dimensional, or mismatched in length.
        opforch.utils.exception.TypeError: If either vector is not uint8 or a signed integer dtype.
        opforch.utils.exception.ValueError: If either vector contains a negative label.

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
    labels: torch.Tensor | list[int],
    preds: torch.Tensor | list[int],
) -> torch.Tensor:
    """Compute one minus each true class's misclassification rate.

    For present classes this is recall, not the false-positive/false-negative average used by opf_accuracy.
    C is one plus the largest true-label ID, and absent true-label classes retain NaN.
    Predicted-only IDs beyond that range do not add output entries.

    Args:
        labels: Nonempty (N,) integer tensor or list of nonnegative true-label IDs.
        preds: (N,) integer tensor or list of nonnegative predicted IDs, moved to the labels' device.

    Returns:
        torch.Tensor: Float64 (C,) scores on the labels' device, in [0, 1] for present classes and NaN otherwise.

    Raises:
        opforch.utils.exception.SizeError: If either vector is empty, not one-dimensional, or mismatched in length.
        opforch.utils.exception.TypeError: If either vector is not uint8 or a signed integer dtype.
        opforch.utils.exception.ValueError: If either vector contains a negative label.

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
    """Compute and save a pairwise score matrix with torch.save.

    Tensor inputs retain their dtype and device during computation, while non-tensors are converted to float32.
    The saved result is moved to CPU. The selected registry function determines the numerical domain and may
    return a divergence or similarity rather than a metric.

    Args:
        data: (N, D) floating tensor or tensor-convertible samples, converted to float32 only for non-tensors.
        output: Destination path for the CPU (N, N) tensor serialized with torch.save.
        distance: Name in opforch.math.distance.DISTANCES selecting the pairwise function.

    Returns:
        None: The matrix is written to output instead of returned.

    Raises:
        KeyError: If distance is absent from the registry.
        RuntimeError: If a PyTorch tensor operation or serialization fails.
        OSError: If an operating-system error prevents opening or writing output.

    """

    logger.info("Precomputing pairwise scores.")

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    dist_fn = d.DISTANCES[distance]
    distances = dist_fn(data, data)

    torch.save(distances.cpu(), output)

    logger.info("Pairwise scores saved to %s.", output)


def purity(
    labels: torch.Tensor | list[int],
    preds: torch.Tensor | list[int],
) -> float:
    """Compute cluster purity by assigning each cluster its majority true label.

    Cluster IDs need not match true-label IDs. Each cluster contributes its largest true-label count,
    and the total is divided by the number of samples.

    Args:
        labels: Nonempty (N,) integer tensor or list of nonnegative true-label IDs.
        preds: (N,) integer tensor or list of nonnegative cluster IDs, moved to the labels' device.

    Returns:
        float: Fraction of samples belonging to their cluster's majority true class, in [0, 1].

    Raises:
        opforch.utils.exception.SizeError: If either vector is empty, not one-dimensional, or mismatched in length.
        opforch.utils.exception.TypeError: If either vector is not uint8 or a signed integer dtype.
        opforch.utils.exception.ValueError: If either vector contains a negative label.

    """

    labels, preds = _label_tensors(labels, preds)
    c_matrix = _confusion_counts(labels, preds, int(labels.max().item()) + 1, int(preds.max().item()) + 1)
    _purity = c_matrix.max(dim=0).values.sum() / len(labels)

    return _purity.item()
