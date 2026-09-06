"""Batched distance metrics operating on 2-D tensors.

Every function takes two tensors X: (N, D) and Y: (M, D) and returns
a distance matrix of shape (N, M). All operations are GPU-compatible.
"""

from typing import Callable, Dict

import torch

import opforch.utils.constants as c

# Alias for the distance function signature
DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Small epsilon to prevent division by zero (added inline, no decorator)
EPS = c.EPSILON


def _expand(X: torch.Tensor, Y: torch.Tensor):
    """Expands X and Y for element-wise broadcasting.

    Args:
        X: (N, D) tensor.
        Y: (M, D) tensor.

    Returns:
        Xe: (N, 1, D), Ye: (1, M, D) ready for broadcasting to (N, M, D).

    """

    return X.unsqueeze(1), Y.unsqueeze(0)


def _expand_safe(X: torch.Tensor, Y: torch.Tensor):
    """Same as _expand but adds EPS to avoid zero-division."""

    return X.unsqueeze(1) + EPS, Y.unsqueeze(0) + EPS


# ---------------------------------------------------------------------------
# Lp-norm family
# ---------------------------------------------------------------------------


def euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Euclidean Distance (L2 Norm)."""

    return torch.cdist(X, Y, p=2)


def squared_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean Distance."""

    return torch.cdist(X, Y, p=2).pow(2)


def manhattan_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Manhattan Distance (L1 Norm)."""

    return torch.cdist(X, Y, p=1)


def chebyshev_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Chebyshev Distance (L∞ Norm)."""

    return torch.cdist(X, Y, p=float("inf"))


def average_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Average Euclidean Distance."""

    sq = squared_euclidean_distance(X, Y)
    return (sq / X.shape[1]).sqrt()


# ---------------------------------------------------------------------------
# Log-transformed distances
# ---------------------------------------------------------------------------


def log_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Log-Euclidean Distance."""

    d = euclidean_distance(X, Y)
    return c.MAX_ARC_WEIGHT * torch.log(d + 1)


def log_squared_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Log-Squared Euclidean Distance (default OPF metric)."""

    d = squared_euclidean_distance(X, Y)
    return c.MAX_ARC_WEIGHT * torch.log(d + 1)


def lorentzian_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Lorentzian Distance."""

    Xe, Ye = _expand(X, Y)
    return torch.log(1 + (Xe - Ye).abs()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Statistical divergences
# ---------------------------------------------------------------------------


def kullback_leibler_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Kullback-Leibler Divergence."""

    Xe, Ye = _expand_safe(X, Y)
    return (Xe * torch.log(Xe / Ye)).sum(dim=-1)


def jeffreys_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Jeffreys Distance (J-Divergence)."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye) * torch.log(Xe / Ye)).sum(dim=-1)


def jensen_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Jensen Distance."""

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    dist = (Xe * torch.log(Xe) + Ye * torch.log(Ye)) / 2 - m * torch.log(m)
    return 0.5 * dist.sum(dim=-1)


def jensen_shannon_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon Distance."""

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    d1 = (Xe * torch.log(Xe / m)).sum(dim=-1)
    d2 = (Ye * torch.log(Ye / m)).sum(dim=-1)
    return 0.5 * (d1 + d2)


def k_divergence_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """K Divergence Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return (Xe * torch.log((2 * Xe) / (Xe + Ye))).sum(dim=-1)


def topsoe_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Topsoe Distance (Information Statistics)."""

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    d1 = (Xe * torch.log(Xe / m)).sum(dim=-1)
    d2 = (Ye * torch.log(Ye / m)).sum(dim=-1)
    return d1 + d2


# ---------------------------------------------------------------------------
# Chi-squared family
# ---------------------------------------------------------------------------


def chi_squared_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Chi-Squared Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return (0.5 * (Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def neyman_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Neyman Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / Xe).sum(dim=-1)


def pearson_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Pearson Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / Ye).sum(dim=-1)


def squared_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Distance (Triangular Discrimination)."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def additive_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Additive Symmetric Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) * (Xe + Ye) / (Xe * Ye)).sum(dim=-1)


def divergence_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Divergence Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) / (Xe + Ye).pow(2)).sum(dim=-1)


def sangvi_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Sangvi Distance (Probabilistic Symmetric)."""

    Xe, Ye = _expand_safe(X, Y)
    return (2 * (Xe - Ye).pow(2) / (Xe + Ye)).sum(dim=-1)


def statistic_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Statistic Distance."""

    Xe, Ye = _expand_safe(X, Y)
    m = (Xe + Ye) / 2
    return ((Xe - m) / m).sum(dim=-1)


# ---------------------------------------------------------------------------
# Set / similarity-based
# ---------------------------------------------------------------------------


def cosine_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Cosine Distance."""

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1).sqrt() * Ye.pow(2).sum(dim=-1).sqrt()
    return 1 - num / den


def dice_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Dice Distance."""

    Xe, Ye = _expand_safe(X, Y)
    num = 2 * (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1) + Ye.pow(2).sum(dim=-1)
    return 1 - num / den


def jaccard_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Jaccard Distance."""

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe - Ye).pow(2).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1) + Ye.pow(2).sum(dim=-1) - (Xe * Ye).sum(dim=-1)
    return num / den


def chord_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Chord Distance."""

    Xe, Ye = _expand_safe(X, Y)
    num = (Xe * Ye).sum(dim=-1)
    den = Xe.pow(2).sum(dim=-1).sqrt() * Ye.pow(2).sum(dim=-1).sqrt()
    return (2 - 2 * num / den).clamp(min=0).sqrt()


def hamming_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Hamming Distance."""

    Xe, Ye = _expand(X, Y)
    return (Xe != Ye).float().sum(dim=-1)


# ---------------------------------------------------------------------------
# Ecological distances
# ---------------------------------------------------------------------------


def bray_curtis_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Bray-Curtis Distance (Sorensen Distance)."""

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / (Xe + Ye).sum(dim=-1)


def canberra_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Canberra Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).abs() / (Xe.abs() + Ye.abs())).sum(dim=-1)


def soergel_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Soergel Distance (Ruzicka Distance)."""

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / torch.maximum(Xe, Ye).sum(dim=-1)


def kulczynski_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Kulczynski Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / torch.minimum(Xe, Ye).sum(dim=-1)


def gower_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Gower Distance (Average Manhattan)."""

    Xe, Ye = _expand(X, Y)
    return (Xe - Ye).abs().sum(dim=-1) / X.shape[1]


# ---------------------------------------------------------------------------
# Probabilistic / distributional
# ---------------------------------------------------------------------------


def bhattacharyya_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return -torch.log((Xe * Ye).sqrt().sum(dim=-1))


def hellinger_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Hellinger Distance (Jeffries-Matusita)."""

    Xe, Ye = _expand(X, Y)
    return (2 * (Xe.sqrt() - Ye.sqrt()).pow(2)).sum(dim=-1).sqrt()


def matusita_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Matusita Distance (features must be positive)."""

    Xe, Ye = _expand(X, Y)
    return (Xe.sqrt() - Ye.sqrt()).pow(2).sum(dim=-1).sqrt()


def squared_chord_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Chord Distance (features must be positive)."""

    Xe, Ye = _expand(X, Y)
    return (Xe.sqrt() - Ye.sqrt()).pow(2).sum(dim=-1)


def hassanat_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Hassanat Distance."""

    Xe, Ye = _expand_safe(X, Y)
    mn = torch.minimum(Xe, Ye)
    mx = torch.maximum(Xe, Ye)

    denominator = 1 + torch.where(mn < 0, mx - mn, mx)
    return (1 - (1 + mn.clamp(min=0)) / denominator).sum(dim=-1)


# ---------------------------------------------------------------------------
# Symmetric / asymmetric
# ---------------------------------------------------------------------------


def max_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Max Symmetric Distance."""

    Xe, Ye = _expand_safe(X, Y)
    d1 = ((Xe - Ye).pow(2) / Xe).sum(dim=-1)
    d2 = ((Xe - Ye).pow(2) / Ye).sum(dim=-1)
    return torch.maximum(d1, d2)


def min_symmetric_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Min Symmetric Distance."""

    Xe, Ye = _expand_safe(X, Y)
    d1 = ((Xe - Ye).pow(2) / Xe).sum(dim=-1)
    d2 = ((Xe - Ye).pow(2) / Ye).sum(dim=-1)
    return torch.minimum(d1, d2)


def vicis_symmetric1_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vicis Symmetric 1 Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.minimum(Xe, Ye).pow(2)).sum(dim=-1)


def vicis_symmetric2_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vicis Symmetric 2 Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.minimum(Xe, Ye)).sum(dim=-1)


def vicis_symmetric3_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vicis Symmetric 3 Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).pow(2) / torch.maximum(Xe, Ye)).sum(dim=-1)


def vicis_wave_hedges_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vicis-Wave Hedges Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye).abs() / torch.minimum(Xe, Ye)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------


def gaussian_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Gaussian Distance (RBF-like, gamma=1)."""

    d = euclidean_distance(X, Y)
    return torch.exp(-d)


def clark_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Clark Distance."""

    Xe, Ye = _expand_safe(X, Y)
    return ((Xe - Ye) / (Xe + Ye).abs()).pow(2).sum(dim=-1).sqrt()


def non_intersection_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Non-Intersection Distance."""

    Xe, Ye = _expand(X, Y)
    return 0.5 * (Xe - Ye).abs().sum(dim=-1)


def mean_censored_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Mean Censored Euclidean Distance."""

    Xe, Ye = _expand_safe(X, Y)
    sq = (Xe - Ye).pow(2).sum(dim=-1)
    nonzero = ((Xe + Ye) != 0).float().sum(dim=-1).clamp(min=1)
    return (sq / nonzero).sqrt()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DISTANCES: Dict[str, DistanceFn] = {
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

# Set of valid distance names (for validation)
VALID_DISTANCES = set(DISTANCES.keys())
