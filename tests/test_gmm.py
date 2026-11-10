import math
import unittest

import torch

from src.gmm import (
    GMMConfig,
    aic,
    average_log_likelihood,
    bic,
    fit_gmm,
    parameter_count,
)
from src.gmm.initialization import initialize_gmm
from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import component_log_probabilities
from src.utils.data_loader import BackgroundRecording, sample_per_speaker, split_speakers


class GMMTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        first = torch.randn((60, 3), generator=generator) * 0.3
        second = torch.randn((60, 3), generator=generator) * 0.3 + 2.0
        self.data = torch.cat((first, second), dim=0)

    def test_all_covariance_types_fit_with_finite_likelihood(self) -> None:
        for covariance_type in ("full", "diagonal", "tied", "spherical"):
            result = fit_gmm(
                self.data,
                GMMConfig(
                    n_components=2,
                    covariance_type=covariance_type,
                    max_iter=30,
                    chunk_size=32,
                ),
                seed=11,
            )
            self.assertTrue(math.isfinite(result.log_likelihood))
            self.assertTrue(
                math.isfinite(average_log_likelihood(self.data, result.model, 32))
            )
            self.assertTrue(torch.allclose(result.model.weights.sum(), torch.tensor(1.0, dtype=torch.float64)))

    def test_information_criteria_parameter_counts_and_values(self) -> None:
        model = initialize_gmm(
            self.data.to(torch.float64),
            n_components=2,
            covariance_type="diagonal",
            reg_covar=1e-4,
            seed=1,
            method="random",
        )
        self.assertEqual(parameter_count(model), (2 - 1) + 2 * 3 + 2 * 3)
        likelihood = -10.0
        self.assertEqual(aic(model, likelihood), 2 * parameter_count(model) + 20)
        self.assertAlmostEqual(
            bic(model, likelihood, 100),
            parameter_count(model) * math.log(100) + 20,
        )

    def test_single_standard_normal_log_probability(self) -> None:
        model = GaussianMixture(
            weights=torch.ones(1, dtype=torch.float64),
            means=torch.zeros((1, 1), dtype=torch.float64),
            covariances=torch.ones((1, 1, 1), dtype=torch.float64),
            covariance_type="full",
            reg_covar=1e-4,
        )
        scores = component_log_probabilities(
            torch.zeros((1, 1), dtype=torch.float64),
            model,
        )
        self.assertAlmostEqual(scores.item(), -0.5 * math.log(2 * math.pi), places=12)

    def test_specialized_covariance_likelihoods_match_gaussian_formula(self) -> None:
        observation = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
        mean = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
        cases = (
            ("diagonal", torch.tensor([[0.5, 2.0]], dtype=torch.float64), [0.5, 2.0]),
            ("spherical", torch.tensor([0.75], dtype=torch.float64), [0.75, 0.75]),
        )
        for covariance_type, covariance, diagonal in cases:
            model = GaussianMixture(
                weights=torch.ones(1, dtype=torch.float64),
                means=mean,
                covariances=covariance,
                covariance_type=covariance_type,
                reg_covar=1e-4,
            )
            difference = observation[0] - mean[0]
            expected = -0.5 * (
                2 * math.log(2 * math.pi)
                + sum(math.log(value) for value in diagonal)
                + sum(
                    float(difference[index] ** 2) / diagonal[index]
                    for index in range(2)
                )
            )
            actual = component_log_probabilities(observation, model).item()
            self.assertAlmostEqual(actual, expected, places=12)

        tied_covariance = torch.tensor(
            [[2.0, 0.25], [0.25, 1.0]],
            dtype=torch.float64,
        )
        tied_model = GaussianMixture(
            weights=torch.ones(1, dtype=torch.float64),
            means=mean,
            covariances=tied_covariance,
            covariance_type="tied",
            reg_covar=1e-4,
        )
        difference = observation[0] - mean[0]
        quadratic = difference @ torch.linalg.solve(tied_covariance, difference)
        expected = -0.5 * (
            2 * math.log(2 * math.pi)
            + torch.linalg.slogdet(tied_covariance).logabsdet.item()
            + quadratic.item()
        )
        actual = component_log_probabilities(observation, tied_model).item()
        self.assertAlmostEqual(actual, expected, places=12)

    def test_random_initialization_is_reproducible(self) -> None:
        config = GMMConfig(2, "diagonal", max_iter=20)
        first = fit_gmm(self.data, config, seed=29)
        second = fit_gmm(self.data, config, seed=29)
        self.assertEqual(first.log_likelihood, second.log_likelihood)
        self.assertTrue(torch.equal(first.model.means, second.model.means))

    def test_kmeans_initializer_is_reproducible(self) -> None:
        first = initialize_gmm(
            self.data.to(torch.float64), 2, "diagonal", 1e-4, 17, "kmeans"
        )
        second = initialize_gmm(
            self.data.to(torch.float64), 2, "diagonal", 1e-4, 17, "kmeans"
        )
        self.assertTrue(torch.equal(first.means, second.means))

    def test_degenerate_covariance_becomes_stable_after_regularization(self) -> None:
        covariance = torch.zeros((3, 3), dtype=torch.float64)
        _, info = torch.linalg.cholesky_ex(covariance)
        self.assertNotEqual(int(info.item()), 0)
        regularized = covariance + 1e-4 * torch.eye(3, dtype=torch.float64)
        _, regularized_info = torch.linalg.cholesky_ex(regularized)
        self.assertEqual(int(regularized_info.item()), 0)

    def test_speaker_split_and_sampling_are_deterministic(self) -> None:
        recordings = [
            BackgroundRecording(
                f"file-{index}",
                f"speaker-{index // 2}",
                torch.full((5, 2), float(index)),
            )
            for index in range(12)
        ]
        fit_speakers, heldout_speakers = split_speakers(recordings, 0.25, 2026)
        fit_again, heldout_again = split_speakers(recordings, 0.25, 2026)
        self.assertFalse(fit_speakers & heldout_speakers)
        self.assertEqual(fit_speakers, fit_again)
        self.assertEqual(heldout_speakers, heldout_again)
        first_sample = sample_per_speaker(recordings, fit_speakers, 3, 2027)
        second_sample = sample_per_speaker(recordings, fit_speakers, 3, 2027)
        self.assertTrue(torch.equal(first_sample, second_sample))
        self.assertEqual(first_sample.shape[0], len(fit_speakers) * 3)


if __name__ == "__main__":
    unittest.main()
