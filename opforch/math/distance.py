# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Pairwise distances, divergences, and similarities for dense tensors.

The shared input contract is X of shape (N, D) and Y of shape (M, D), with matching floating dtypes and devices.
Outputs have shape (N, M) on the input device, and inputs are not modified. Device and dtype support follow
the PyTorch operations used by each function. Result dtypes follow PyTorch promotion rules, except that
Hamming explicitly returns float32. Elementwise implementations broadcast (N, 1, D) and (1, M, D) operands.

Selected functions add EPS to each operand as numerical protection. Their domain notes use "shifted" to mean
values after this addition. EPS can underflow in low precision and does not validate inputs, guarantee finite
results, or normalize rows into probability distributions. Domain restrictions describe meaningful real-valued
evaluation, not additional runtime checks. Invalid logarithms, square roots, divisions, or overflow may produce
NaN or infinity instead of raising an exception.

DISTANCES maps registry names to callables, and VALID_DISTANCES contains those names. The registry includes
directional divergences and a similarity, so names do not imply symmetry or the triangle inequality.

"""

from collections.abc import Callable

import torch

import opforch.utils.constants as c

DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

EPS = c.EPSILON


def _expand(X: torch.Tensor, Y: torch.Tensor):
    return X.unsqueeze(1), Y.unsqueeze(0)


def _expand_safe(X: torch.Tensor, Y: torch.Tensor):
    return X.unsqueeze(1) + EPS, Y.unsqueeze(0) + EPS


def euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean norm of each pairwise row difference.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Euclidean distances of shape (N, M) on X's device.

    """

    return torch.cdist(X, Y, p=2)


def squared_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Square the Euclidean norm of each pairwise row difference.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Squared Euclidean distances of shape (N, M) on X's device.

    """

    return torch.cdist(X, Y, p=2).pow(2)


def manhattan_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum the absolute coordinate differences for each pair of rows.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Manhattan distances of shape (N, M) on X's device.

    """

    return torch.cdist(X, Y, p=1)


def chebyshev_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Take the largest absolute coordinate difference for each pair of rows.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Chebyshev distances of shape (N, M) on X's device.

    """

    return torch.cdist(X, Y, p=float("inf"))


def average_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the root mean squared coordinate difference for each pair of rows.

    The Euclidean norm is divided by sqrt(D), not D. A positive feature count D is needed for a defined mean.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Root mean squared differences of shape (N, M) on X's device.

    """

    sq = squared_euclidean_distance(X, Y)
    return (sq / X.shape[1]).sqrt()


def log_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the scaled natural logarithm of one plus each Euclidean distance.

    The result is ``MAX_ARC_WEIGHT * log(1 + ||X - Y||_2)``, using the scale from opforch.utils.constants.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Scaled log-Euclidean scores of shape (N, M) on X's device.

    """

    d = euclidean_distance(X, Y)
    return c.MAX_ARC_WEIGHT * torch.log(d + 1)


def log_squared_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the scaled natural logarithm of one plus each squared Euclidean distance.

    The result is ``MAX_ARC_WEIGHT * log(1 + ||X - Y||_2 ** 2)``, using the scale from opforch.utils.constants.
    This is the default registry entry used for OPF distances.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Scaled log-squared-Euclidean scores of shape (N, M) on X's device.

    """

    d = squared_euclidean_distance(X, Y)
    return c.MAX_ARC_WEIGHT * torch.log(d + 1)


def lorentzian_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum the natural logarithms of one plus each absolute coordinate difference.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Lorentzian scores of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return torch.log(1 + (Xe - Ye).abs()).sum(dim=-1)


def kullback_leibler_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum shifted X coordinates weighted by their log ratios against shifted Y.

    This computes the directional expression ``sum(a * log(a / b))`` for EPS-shifted a and b.
    Nonnegative histogram features are intended. Logarithm arguments must be positive, with nonzero shifted Y entries.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Directed Kullback-Leibler expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (Xe * torch.log(Xe / Ye)).sum(dim=-1)


def jeffreys_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum shifted coordinate differences multiplied by their log ratios.

    This uses ``sum((a - b) * log(a / b))`` for EPS-shifted a and b without probability normalization.
    Logarithm arguments must be positive, with nonzero shifted Y entries, as for nonnegative histogram features.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Jeffreys expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye) * torch.log(Xe / Ye)).sum(dim=-1)


def jensen_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Jensen entropy difference with its historical extra 0.5 factor.

    For shifted a and b with midpoint m, the sum of ``(a * log(a) + b * log(b)) / 2 - m * log(m)``
    is multiplied by 0.5. Each shifted value and midpoint must be positive for the logarithms.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Half-scaled Jensen expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    dist = (Xe * torch.log(Xe) + Ye * torch.log(Ye)) / 2 - m * torch.log(m)
    return 0.5 * dist.sum(dim=-1)


def jensen_shannon_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence without taking its square root.

    The result averages two directed log-ratio sums against the EPS-shifted midpoint.
    Shifted midpoint entries must be nonzero and both logarithm arguments must be positive.
    Nonnegative histogram or probability rows provide the intended interpretation.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Jensen-Shannon expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    d1 = (Xe * torch.log(Xe / m)).sum(dim=-1)
    d2 = (Ye * torch.log(Ye / m)).sum(dim=-1)
    return 0.5 * (d1 + d2)


def k_divergence_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the directed log-ratio score against each coordinatewise midpoint.

    This sums ``a * log(2 * a / (a + b))`` for EPS-shifted a and b without probability normalization.
    Shifted sums must be nonzero and logarithm arguments must be positive.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Directed K-divergence expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (Xe * torch.log((2 * Xe) / (Xe + Ye))).sum(dim=-1)


def topsoe_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum both directed log-ratio scores against their coordinatewise midpoint.

    This adds the two terms rather than averaging them as in jensen_shannon_distance.
    Shifted midpoint entries must be nonzero and both logarithm arguments must be positive.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Topsoe expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    d1 = (Xe * torch.log(Xe / m)).sum(dim=-1)
    d2 = (Ye * torch.log(Ye / m)).sum(dim=-1)
    return d1 + d2


def chi_squared_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute half the sum of squared differences divided by coordinate sums.

    Uses EPS-shifted operands. Nonnegative features are intended, and shifted pairwise sums must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Half-scaled chi-squared expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (0.5 * (Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def neyman_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared shifted differences divided by shifted X coordinates.

    Shifted X entries must be nonzero. Nonnegative features provide the intended chi-squared interpretation.

    Args:
        X: Floating tensor of shape (N, D), providing the denominators after adding EPS.
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Directed Neyman expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / Xe).sum(dim=-1)


def pearson_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared shifted differences divided by shifted Y coordinates.

    Shifted Y entries must be nonzero. Nonnegative features provide the intended chi-squared interpretation.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device, providing denominators after adding EPS.

    Returns:
        torch.Tensor: Directed Pearson expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / Ye).sum(dim=-1)


def squared_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the triangular-discrimination expression from EPS-shifted operands.

    Squared differences are divided by coordinatewise sums, which must be nonzero.
    Nonnegative features provide the intended interpretation.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Triangular-discrimination expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def additive_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum twice the squared differences times the coordinate sum divided by the product.

    Both EPS-shifted operands must have nonzero coordinates to avoid a zero product in the denominator.
    Nonnegative features provide the intended chi-squared interpretation.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Additive symmetric expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) * (Xe + Ye) / (Xe * Ye)).sum(dim=-1)


def divergence_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum twice the squared differences divided by squared coordinate sums.

    Shifted pairwise sums must be nonzero. Unlike the triangular form, each denominator is squared.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Divergence expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) / (Xe + Ye).pow(2)).sum(dim=-1)


def sangvi_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute twice the triangular-discrimination expression.

    Uses EPS-shifted operands. Nonnegative features are intended, and shifted coordinatewise sums must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Sangvi expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def statistic_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the signed sum of deviations from each coordinatewise midpoint.

    Uses shifted a and b with ``m = (a + b) / 2`` and sums ``(a - m) / m``.
    Midpoints must be nonzero. The numerator is neither squared nor absolute, so scores can be negative.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Signed statistic expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    return ((Xe - m) / m).sum(dim=-1)


def cosine_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute one minus cosine similarity after adding EPS to every coordinate.

    Both shifted row norms must be nonzero. No clipping is applied, so rounding may affect the theoretical range.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Shifted cosine dissimilarities of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1).sqrt() * Ye.pow(2).sum(dim=-1).sqrt()
    return 1 - num / den


def dice_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute one minus the continuous Dice similarity of EPS-shifted rows.

    The similarity is twice the dot product divided by the sum of squared row norms, not a binary-set comparison.
    The sum of shifted squared norms must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Continuous Dice dissimilarities of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    num = 2 * (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1) + Ye.pow(2).sum(dim=-1)
    return 1 - num / den


def jaccard_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Divide squared row differences by the continuous Jaccard denominator.

    The denominator is the sum of shifted squared row norms minus their dot product and must be nonzero.
    This is not a comparison of set cardinalities.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Continuous Jaccard expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe - Ye).pow(2).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1) + Ye.pow(2).sum(dim=-1) - (Xe * Ye).sum(dim=-1)
    return num / den


def chord_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the square root of twice the shifted cosine dissimilarity.

    Both shifted row norms must be nonzero. The square-root argument is clamped to at least zero to protect
    against rounding below zero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Shifted chord expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1).sqrt() * Ye.pow(2).sum(dim=-1).sqrt()
    return (2 - 2 * num / den).clamp(min=0).sqrt()


def hamming_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Count unequal coordinates for each pair of rows.

    Equality is exact, without a tolerance. The result is a count, not the fraction of unequal coordinates.
    This function explicitly converts the comparison mask to float32 before summing.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Float32 mismatch counts of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return (Xe != Ye).float().sum(dim=-1)


def bray_curtis_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Divide summed absolute differences by the signed sum of shifted operands.

    The total shifted sum must be nonzero. Nonnegative features provide the intended Bray-Curtis form,
    because the denominator is not an absolute sum.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Bray-Curtis expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / (Xe + Ye).sum(dim=-1)


def canberra_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum absolute differences divided by sums of absolute shifted coordinates.

    Each coordinatewise sum of shifted absolute values must be nonzero.
    The absolute denominator allows signed input features.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Canberra expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).abs() / (Xe.abs() + Ye.abs())).sum(dim=-1)


def soergel_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Divide summed absolute differences by the sum of coordinatewise maxima.

    Uses EPS-shifted operands. Nonnegative features are intended, and the sum of shifted maxima must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Soergel expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / torch.maximum(Xe, Ye).sum(dim=-1)


def kulczynski_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Divide summed absolute differences by the sum of coordinatewise minima.

    Uses EPS-shifted operands. Nonnegative features are intended, and the sum of shifted minima must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Kulczynski expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / torch.minimum(Xe, Ye).sum(dim=-1)


def gower_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the mean absolute coordinate difference for each pair of rows.

    This implementation is mean Manhattan distance, not mixed-type, range-normalized Gower distance.
    A positive feature count D is needed for a defined mean, and no per-feature ranges are used.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Mean Manhattan distances of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / X.shape[1]


def bhattacharyya_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Take the negative logarithm of summed square roots of shifted coordinate products.

    Shifted products must be nonnegative, and their square-root sum must be positive.
    Rows are not probability-normalized, so the result can be negative when that sum exceeds one.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Bhattacharyya expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return -torch.log((Xe * Ye).sqrt().sum(dim=-1))


def hellinger_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the square-root-coordinate distance with historical Hellinger scaling.

    The result is ``sqrt(2 * sum((sqrt(X) - sqrt(Y)) ** 2))``, not the usual unit-probability scaling.
    Raw coordinates must be nonnegative. No EPS is added, and rows are not normalized.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Historically scaled Hellinger expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return (2 * (Xe.sqrt() - Ye.sqrt()).pow(2)).sum(dim=-1).sqrt()


def matusita_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean norm of differences between square-root coordinates.

    Raw coordinates must be nonnegative, including zero. No EPS is added, and rows are not normalized.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Matusita expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return (Xe.sqrt() - Ye.sqrt()).pow(2).sum(dim=-1).sqrt()


def squared_chord_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared differences between square-root coordinates without an outer square root.

    Raw coordinates must be nonnegative, including zero. No EPS is added, and rows are not normalized.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Squared-chord expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return (Xe.sqrt() - Ye.sqrt()).pow(2).sum(dim=-1)


def hassanat_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum bounded coordinate contributions using the signed-feature Hassanat branches.

    Uses EPS-shifted coordinatewise minima and maxima. A negative minimum selects the denominator
    ``1 + maximum - minimum``, otherwise it is ``1 + maximum``. The numerator is ``1 + max(minimum, 0)``.
    Signed features are supported without requiring nonnegative inputs.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Hassanat expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    mn = torch.minimum(Xe, Ye)
    mx = torch.maximum(Xe, Ye)

    denominator = 1 + torch.where(mn < 0, mx - mn, mx)
    return (1 - (1 + mn.clamp(min=0)) / denominator).sum(dim=-1)


def max_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Take the larger of two directed shifted chi-squared scores.

    The directional terms divide squared differences by X + EPS and Y + EPS, respectively.
    Both shifted operands must have nonzero coordinates.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Maximum directed scores of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    d1 = ((Xe - Ye).pow(2) / Xe).sum(dim=-1)
    d2 = ((Xe - Ye).pow(2) / Ye).sum(dim=-1)
    return torch.maximum(d1, d2)


def min_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Take the smaller of two directed shifted chi-squared scores.

    The directional terms divide squared differences by X + EPS and Y + EPS, respectively.
    Both shifted operands must have nonzero coordinates.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Minimum directed scores of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    d1 = ((Xe - Ye).pow(2) / Xe).sum(dim=-1)
    d2 = ((Xe - Ye).pow(2) / Ye).sum(dim=-1)
    return torch.minimum(d1, d2)


def vicis_symmetric1_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared differences divided by squared coordinatewise minima.

    Uses EPS-shifted operands, whose coordinatewise minima must be nonzero.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: First-variant Vicis expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.minimum(Xe, Ye).pow(2)).sum(dim=-1)


def vicis_symmetric2_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared differences divided by coordinatewise minima.

    The shifted minima must be nonzero. Unlike variant 1, the denominator is not squared,
    so signed inputs can give negative scores.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Second-variant Vicis expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.minimum(Xe, Ye)).sum(dim=-1)


def vicis_symmetric3_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum squared differences divided by coordinatewise maxima.

    The shifted maxima must be nonzero. The denominator is not squared, so negative denominators
    can give negative scores.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Third-variant Vicis expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.maximum(Xe, Ye)).sum(dim=-1)


def vicis_wave_hedges_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sum absolute differences divided by coordinatewise minima.

    Uses EPS-shifted operands. The shifted minima must be nonzero, and negative minima can give negative scores.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Vicis-Wave Hedges expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).abs() / torch.minimum(Xe, Ye)).sum(dim=-1)


def gaussian_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Return exp(-Euclidean distance) as a pairwise similarity.

    Despite the registry name, this is a similarity, not a zero-diagonal distance.
    Equal rows have value one in exact arithmetic, and the exponent uses the Euclidean norm rather than its square.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Exponentiated negative Euclidean norms of shape (N, M) on X's device.

    """

    d = euclidean_distance(X, Y)
    return torch.exp(-d)


def clark_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean norm of differences divided by absolute coordinate sums.

    Uses EPS-shifted operands, whose coordinatewise sums must be nonzero before taking their absolute values.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Clark expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye) / (Xe + Ye).abs()).pow(2).sum(dim=-1).sqrt()


def non_intersection_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute half the Manhattan distance without adding EPS.

    The factor 0.5 applies to arbitrary input rows, which are not probability-normalized.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Half-Manhattan distances of shape (N, M) on X's device.

    """

    Xe, Ye = _expand(X, Y)
    return 0.5 * (Xe - Ye).abs().sum(dim=-1)


def mean_censored_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Scale Euclidean differences using counts of nonzero shifted coordinate sums.

    All shifted squared differences contribute to the numerator. The denominator counts coordinates where
    ``(X + EPS) + (Y + EPS)`` is nonzero, is clamped to at least one, and divides the sum before its square root.
    This counts shifted sums, not simply nonzero original features. Its float32 count can promote low-precision inputs.

    Args:
        X: Floating tensor of shape (N, D).
        Y: Floating tensor of shape (M, D) with X's dtype and device.

    Returns:
        torch.Tensor: Count-scaled Euclidean expressions of shape (N, M) on X's device.

    """

    Xe, Ye = _expand_safe(X, Y)
    sq = (Xe - Ye).pow(2).sum(dim=-1)
    nonzero = ((Xe + Ye) != 0).float().sum(dim=-1).clamp(min=1)
    return (sq / nonzero).sqrt()


DISTANCES: dict[str, DistanceFn] = {
    "additive_symmetric": additive_symmetric_distance,
    "average_euclidean": average_euclidean_distance,
    "bhattacharyya": bhattacharyya_distance,
    "bray_curtis": bray_curtis_distance,
    "canberra": canberra_distance,
    "chebyshev": chebyshev_distance,
    "chi_squared": chi_squared_distance,
    "chord": chord_distance,
    "clark": clark_distance,
    "cosine": cosine_distance,
    "dice": dice_distance,
    "divergence": divergence_distance,
    "euclidean": euclidean_distance,
    "gaussian": gaussian_distance,
    "gower": gower_distance,
    "hamming": hamming_distance,
    "hassanat": hassanat_distance,
    "hellinger": hellinger_distance,
    "jaccard": jaccard_distance,
    "jeffreys": jeffreys_distance,
    "jensen": jensen_distance,
    "jensen_shannon": jensen_shannon_distance,
    "k_divergence": k_divergence_distance,
    "kulczynski": kulczynski_distance,
    "kullback_leibler": kullback_leibler_distance,
    "log_euclidean": log_euclidean_distance,
    "log_squared_euclidean": log_squared_euclidean_distance,
    "lorentzian": lorentzian_distance,
    "manhattan": manhattan_distance,
    "matusita": matusita_distance,
    "max_symmetric": max_symmetric_distance,
    "mean_censored_euclidean": mean_censored_euclidean_distance,
    "min_symmetric": min_symmetric_distance,
    "neyman": neyman_distance,
    "non_intersection": non_intersection_distance,
    "pearson": pearson_distance,
    "sangvi": sangvi_distance,
    "soergel": soergel_distance,
    "squared": squared_distance,
    "squared_chord": squared_chord_distance,
    "squared_euclidean": squared_euclidean_distance,
    "statistic": statistic_distance,
    "topsoe": topsoe_distance,
    "vicis_symmetric1": vicis_symmetric1_distance,
    "vicis_symmetric2": vicis_symmetric2_distance,
    "vicis_symmetric3": vicis_symmetric3_distance,
    "vicis_wave_hedges": vicis_wave_hedges_distance,
}

VALID_DISTANCES = set(DISTANCES.keys())
