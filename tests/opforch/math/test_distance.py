# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch

from opforch.math import distance


def _t(values):
    return torch.tensor([values], dtype=torch.float32)


x = _t([5.1, 3.5, 1.4, 0.3])
y = _t([5.4, 3.4, 1.7, 0.2])


def test_additive_symmetric_distance_returns_positive_pairwise_cost():
    dist = distance.additive_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_average_euclidean_distance_returns_positive_pairwise_cost():
    dist = distance.average_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_bhattacharyya_distance_returns_pairwise_shape():
    dist = distance.bhattacharyya_distance(x, y)
    assert dist.shape == (1, 1)


def test_bray_curtis_distance_returns_nonnegative_pairwise_cost():
    dist = distance.bray_curtis_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_canberra_distance_returns_positive_pairwise_cost():
    dist = distance.canberra_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_chebyshev_distance_matches_known_value():
    dist = distance.chebyshev_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.3) < 0.01


def test_chi_squared_distance_returns_nonnegative_pairwise_cost():
    dist = distance.chi_squared_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_chord_distance_returns_nonnegative_pairwise_cost():
    dist = distance.chord_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_clark_distance_returns_positive_pairwise_cost():
    dist = distance.clark_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_cosine_distance_returns_nonnegative_pairwise_cost():
    dist = distance.cosine_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_dice_distance_returns_nonnegative_pairwise_cost():
    dist = distance.dice_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_divergence_distance_returns_nonnegative_pairwise_cost():
    dist = distance.divergence_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_euclidean_distance_matches_known_value():
    dist = distance.euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.4472) < 0.01


def test_gaussian_distance_returns_positive_pairwise_cost():
    dist = distance.gaussian_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_gower_distance_returns_positive_pairwise_cost():
    dist = distance.gower_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_hamming_distance_counts_unequal_features():
    dist = distance.hamming_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() == 4


def test_hassanat_distance_returns_positive_pairwise_cost():
    dist = distance.hassanat_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_hassanat_distance_handles_negative_features():
    left = torch.tensor([[-2.0], [-1.0], [0.0]])
    right = torch.tensor([[-1.0], [1.0]])

    actual = distance.hassanat_distance(left, right)

    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.75], [0.0, 2 / 3], [0.5, 0.5]]))

    for dtype in (torch.float32, torch.float64):
        identical = torch.tensor([[-1e20]], dtype=dtype)
        torch.testing.assert_close(
            distance.hassanat_distance(identical, identical),
            torch.zeros(1, 1, dtype=dtype),
        )


def test_hellinger_distance_returns_positive_pairwise_cost():
    dist = distance.hellinger_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_jaccard_distance_returns_nonnegative_pairwise_cost():
    dist = distance.jaccard_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_jeffreys_distance_returns_pairwise_shape():
    dist = distance.jeffreys_distance(x, y)
    assert dist.shape == (1, 1)


def test_jensen_distance_returns_pairwise_shape():
    dist = distance.jensen_distance(x, y)
    assert dist.shape == (1, 1)


def test_jensen_shannon_distance_returns_pairwise_shape():
    dist = distance.jensen_shannon_distance(x, y)
    assert dist.shape == (1, 1)


def test_k_divergence_distance_returns_pairwise_shape():
    dist = distance.k_divergence_distance(x, y)
    assert dist.shape == (1, 1)


def test_kulczynski_distance_returns_positive_pairwise_cost():
    dist = distance.kulczynski_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_kullback_leibler_distance_returns_pairwise_shape():
    dist = distance.kullback_leibler_distance(x, y)
    assert dist.shape == (1, 1)


def test_log_euclidean_distance_returns_positive_pairwise_cost():
    dist = distance.log_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_log_squared_euclidean_distance_returns_positive_pairwise_cost():
    dist = distance.log_squared_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_lorentzian_distance_returns_positive_pairwise_cost():
    dist = distance.lorentzian_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_manhattan_distance_matches_known_value():
    dist = distance.manhattan_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.8) < 0.01


def test_matusita_distance_returns_positive_pairwise_cost():
    dist = distance.matusita_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_max_symmetric_distance_returns_positive_pairwise_cost():
    dist = distance.max_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_mean_censored_euclidean_distance_returns_positive_pairwise_cost():
    dist = distance.mean_censored_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_min_symmetric_distance_returns_positive_pairwise_cost():
    dist = distance.min_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_neyman_distance_returns_positive_pairwise_cost():
    dist = distance.neyman_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_non_intersection_distance_returns_positive_pairwise_cost():
    dist = distance.non_intersection_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_pearson_distance_returns_positive_pairwise_cost():
    dist = distance.pearson_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_sangvi_distance_returns_positive_pairwise_cost():
    dist = distance.sangvi_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_soergel_distance_returns_positive_pairwise_cost():
    dist = distance.soergel_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_distance_returns_positive_pairwise_cost():
    dist = distance.squared_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_chord_distance_returns_positive_pairwise_cost():
    dist = distance.squared_chord_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_euclidean_distance_matches_known_value():
    dist = distance.squared_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.2) < 0.01


def test_statistic_distance_returns_pairwise_shape():
    dist = distance.statistic_distance(x, y)
    assert dist.shape == (1, 1)


def test_topsoe_distance_returns_pairwise_shape():
    dist = distance.topsoe_distance(x, y)
    assert dist.shape == (1, 1)


def test_vicis_symmetric1_distance_returns_positive_pairwise_cost():
    dist = distance.vicis_symmetric1_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_symmetric2_distance_returns_positive_pairwise_cost():
    dist = distance.vicis_symmetric2_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_symmetric3_distance_returns_positive_pairwise_cost():
    dist = distance.vicis_symmetric3_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_wave_hedges_distance_returns_positive_pairwise_cost():
    dist = distance.vicis_wave_hedges_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_euclidean_distance_supports_batched_inputs():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    Y = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)

    dist = distance.euclidean_distance(X, Y)
    assert dist.shape == (2, 3)


def test_valid_distances_match_registered_functions():
    assert len(distance.VALID_DISTANCES) == len(distance.DISTANCES)
    assert "euclidean" in distance.VALID_DISTANCES
    assert "log_squared_euclidean" in distance.VALID_DISTANCES
