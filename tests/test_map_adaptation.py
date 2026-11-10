import unittest

import torch

from src.gmm.gmm import GaussianMixture
from src.gmm.map_adaptation import (
    accumulate_sufficient_statistics,
    adapt_means_map,
)


class MAPAdaptationTests(unittest.TestCase):
    def test_statistics_and_mean_adaptation_match_expected_formula(self) -> None:
        ubm = GaussianMixture(
            weights=torch.tensor([0.6, 0.4], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            covariances=torch.ones((2, 2), dtype=torch.float64),
            covariance_type="diagonal",
            reg_covar=1e-4,
        )
        observations = torch.tensor(
            [[0.0, 0.2], [0.1, 0.0], [1.0, 0.9]],
            dtype=torch.float64,
        )
        stats = accumulate_sufficient_statistics(observations, ubm, chunk_size=2)
        result = adapt_means_map(observations, ubm, relevance_factor=8, chunk_size=2)
        expected = (
            stats.first_order + 8 * ubm.means
        ) / (
            stats.occupancies[:, None] + 8
        )
        self.assertTrue(torch.allclose(result.model.means, expected))
        self.assertTrue(torch.allclose(stats.occupancies.sum(), torch.tensor(3.0, dtype=torch.float64)))
        self.assertTrue(torch.equal(result.model.weights, ubm.weights))
        self.assertTrue(torch.equal(result.model.covariances, ubm.covariances))
        self.assertTrue(torch.isfinite(result.model.means).all())

    def test_zero_occupancy_retains_exact_ubm_mean(self) -> None:
        ubm = GaussianMixture(
            weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
            means=torch.tensor([[0.0], [1000.0]], dtype=torch.float64),
            covariances=torch.ones((2, 1), dtype=torch.float64),
            covariance_type="diagonal",
            reg_covar=1e-4,
        )
        observations = torch.tensor([[0.0], [0.1]], dtype=torch.float64)
        result = adapt_means_map(observations, ubm, relevance_factor=4)
        self.assertEqual(result.statistics.occupancies[1].item(), 0.0)
        self.assertEqual(result.model.means[1].item(), ubm.means[1].item())
        self.assertEqual(result.retained_components, 1)

    def test_relevance_factor_limits(self) -> None:
        ubm = GaussianMixture(
            weights=torch.ones(1, dtype=torch.float64),
            means=torch.tensor([[2.0]], dtype=torch.float64),
            covariances=torch.ones((1, 1), dtype=torch.float64),
            covariance_type="diagonal",
            reg_covar=1e-4,
        )
        observations = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        near_zero = adapt_means_map(
            observations,
            ubm,
            relevance_factor=1e-9,
            occupancy_epsilon=0,
        )
        near_infinity = adapt_means_map(
            observations,
            ubm,
            relevance_factor=1e12,
        )
        self.assertAlmostEqual(near_zero.model.means.item(), 0.5, places=8)
        self.assertAlmostEqual(near_infinity.model.means.item(), 2.0, places=10)

    def test_rejects_empty_or_nonfinite_enrolment_features(self) -> None:
        ubm = GaussianMixture(
            weights=torch.ones(1, dtype=torch.float64),
            means=torch.zeros((1, 2), dtype=torch.float64),
            covariances=torch.ones((1, 2), dtype=torch.float64),
            covariance_type="diagonal",
            reg_covar=1e-4,
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            adapt_means_map(torch.empty((0, 2), dtype=torch.float64), ubm, 4)
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            adapt_means_map(
                torch.tensor([[float("nan"), 0.0]], dtype=torch.float64),
                ubm,
                4,
            )


if __name__ == "__main__":
    unittest.main()
