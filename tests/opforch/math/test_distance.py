import torch

from opforch.math import distance


def _t(values):
    """Helper to create a 2-D tensor from a list."""
    return torch.tensor([values], dtype=torch.float32)


x = _t([5.1, 3.5, 1.4, 0.3])
y = _t([5.4, 3.4, 1.7, 0.2])


def test_additive_symmetric_distance():
    dist = distance.additive_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_average_euclidean_distance():
    dist = distance.average_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_bhattacharyya_distance():
    dist = distance.bhattacharyya_distance(x, y)
    assert dist.shape == (1, 1)


def test_bray_curtis_distance():
    dist = distance.bray_curtis_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_canberra_distance():
    dist = distance.canberra_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_chebyshev_distance():
    dist = distance.chebyshev_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.3) < 0.01


def test_chi_squared_distance():
    dist = distance.chi_squared_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_chord_distance():
    dist = distance.chord_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_clark_distance():
    dist = distance.clark_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_cosine_distance():
    dist = distance.cosine_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_dice_distance():
    dist = distance.dice_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_divergence_distance():
    dist = distance.divergence_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_euclidean_distance():
    dist = distance.euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.4472) < 0.01


def test_gaussian_distance():
    dist = distance.gaussian_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_gower_distance():
    dist = distance.gower_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_hamming_distance():
    dist = distance.hamming_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() == 4


def test_hassanat_distance():
    dist = distance.hassanat_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_hassanat_distance_with_negative_features():
    left = torch.tensor([[-2.0], [-1.0], [0.0]])
    right = torch.tensor([[-1.0], [1.0]])

    actual = distance.hassanat_distance(left, right)

    torch.testing.assert_close(
        actual, torch.tensor([[0.5, 0.75], [0.0, 2 / 3], [0.5, 0.5]])
    )

    for dtype in (torch.float32, torch.float64):
        identical = torch.tensor([[-1e20]], dtype=dtype)
        torch.testing.assert_close(
            distance.hassanat_distance(identical, identical),
            torch.zeros(1, 1, dtype=dtype),
        )


def test_hellinger_distance():
    dist = distance.hellinger_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_jaccard_distance():
    dist = distance.jaccard_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() >= 0


def test_jeffreys_distance():
    dist = distance.jeffreys_distance(x, y)
    assert dist.shape == (1, 1)


def test_jensen_distance():
    dist = distance.jensen_distance(x, y)
    assert dist.shape == (1, 1)


def test_jensen_shannon_distance():
    dist = distance.jensen_shannon_distance(x, y)
    assert dist.shape == (1, 1)


def test_k_divergence_distance():
    dist = distance.k_divergence_distance(x, y)
    assert dist.shape == (1, 1)


def test_kulczynski_distance():
    dist = distance.kulczynski_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_kullback_leibler_distance():
    dist = distance.kullback_leibler_distance(x, y)
    assert dist.shape == (1, 1)


def test_log_euclidean_distance():
    dist = distance.log_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_log_squared_euclidean_distance():
    dist = distance.log_squared_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_lorentzian_distance():
    dist = distance.lorentzian_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_manhattan_distance():
    dist = distance.manhattan_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.8) < 0.01


def test_matusita_distance():
    dist = distance.matusita_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_max_symmetric_distance():
    dist = distance.max_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_mean_censored_euclidean_distance():
    dist = distance.mean_censored_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_min_symmetric_distance():
    dist = distance.min_symmetric_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_neyman_distance():
    dist = distance.neyman_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_non_intersection_distance():
    dist = distance.non_intersection_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_pearson_distance():
    dist = distance.pearson_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_sangvi_distance():
    dist = distance.sangvi_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_soergel_distance():
    dist = distance.soergel_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_distance():
    dist = distance.squared_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_chord_distance():
    dist = distance.squared_chord_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_squared_euclidean_distance():
    dist = distance.squared_euclidean_distance(x, y)
    assert dist.shape == (1, 1)
    assert abs(dist.item() - 0.2) < 0.01


def test_statistic_distance():
    dist = distance.statistic_distance(x, y)
    assert dist.shape == (1, 1)


def test_topsoe_distance():
    dist = distance.topsoe_distance(x, y)
    assert dist.shape == (1, 1)


def test_vicis_symmetric1_distance():
    dist = distance.vicis_symmetric1_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_symmetric2_distance():
    dist = distance.vicis_symmetric2_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_symmetric3_distance():
    dist = distance.vicis_symmetric3_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_vicis_wave_hedges_distance():
    dist = distance.vicis_wave_hedges_distance(x, y)
    assert dist.shape == (1, 1)
    assert dist.item() > 0


def test_batched_distance():
    """Test that distance functions work with batched inputs."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    Y = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)

    dist = distance.euclidean_distance(X, Y)
    assert dist.shape == (2, 3)


def test_valid_distances():
    assert len(distance.VALID_DISTANCES) == len(distance.DISTANCES)
    assert "euclidean" in distance.VALID_DISTANCES
    assert "log_squared_euclidean" in distance.VALID_DISTANCES
