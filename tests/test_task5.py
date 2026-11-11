import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.task5.data import Trial, load_trials_and_features
from experiments.task5.run_all import _validate_submission
from experiments.task5.scoring import (
    development_curve,
    operating_rates,
    score_trials,
)
from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import component_log_probabilities


def _single_component_model(mean: float, variance: float = 1.0) -> GaussianMixture:
    return GaussianMixture(
        weights=torch.ones(1, dtype=torch.float64),
        means=torch.full((1, 1), mean, dtype=torch.float64),
        covariances=torch.full((1, 1), variance, dtype=torch.float64),
        covariance_type="diagonal",
        reg_covar=1e-4,
    )


class Task5ScoringTests(unittest.TestCase):
    def test_score_is_average_per_frame_model_llr(self) -> None:
        features = torch.tensor([[0.5], [1.5]], dtype=torch.float64)
        speaker = _single_component_model(1.0)
        ubm = _single_component_model(0.0)
        trial = Trial("t1", "speaker-a", "utt-a", 1)

        result = score_trials(
            [trial],
            {"utt-a": features},
            {"speaker-a": speaker},
            ubm,
        )[0]
        speaker_ll = torch.logsumexp(
            component_log_probabilities(features, speaker),
            dim=1,
        )
        ubm_ll = torch.logsumexp(component_log_probabilities(features, ubm), dim=1)
        expected = float((speaker_ll - ubm_ll).mean().item())
        self.assertAlmostEqual(float(result["score"]), expected, places=12)

    def test_claimed_id_selects_its_own_adapted_model(self) -> None:
        features = torch.tensor([[1.0]], dtype=torch.float64)
        models = {
            "speaker-a": _single_component_model(1.0),
            "speaker-b": _single_component_model(-1.0),
        }
        trials = [
            Trial("t1", "speaker-a", "utt", None),
            Trial("t2", "speaker-b", "utt", None),
        ]
        results = score_trials(
            trials,
            {"utt": features},
            models,
            _single_component_model(0.0),
        )
        self.assertGreater(float(results[0]["score"]), float(results[1]["score"]))
        self.assertNotIn("label", results[0])

    def test_score_rejects_missing_claimed_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "No adapted model"):
            score_trials(
                [Trial("t1", "missing", "utt")],
                {"utt": torch.zeros((1, 1), dtype=torch.float64)},
                {},
                _single_component_model(0.0),
            )

    def test_development_curve_treats_equal_scores_as_one_threshold(self) -> None:
        scored = [
            {"label": 1, "score": 1.0},
            {"label": 0, "score": 1.0},
            {"label": 1, "score": 0.0},
            {"label": 0, "score": 0.0},
        ]
        metrics, curve = development_curve(scored)
        self.assertEqual(metrics["eer"], 0.5)
        self.assertEqual(metrics["far_at_threshold"], 0.5)
        self.assertEqual(metrics["frr_at_threshold"], 0.5)
        far, frr = operating_rates(scored, float(metrics["threshold"]))
        self.assertEqual((far, frr), (0.5, 0.5))
        self.assertEqual(len(curve), 3)

    def test_development_curve_has_zero_eer_for_separated_scores(self) -> None:
        scored = [
            {"label": 1, "score": 2.0},
            {"label": 0, "score": 0.0},
        ]
        metrics, _ = development_curve(scored)
        self.assertEqual(metrics["eer"], 0.0)
        self.assertTrue(math.isfinite(float(metrics["threshold"])))

    def test_load_trials_caches_features_and_does_not_read_eval_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            feature_dir = root / "features_output" / "eval" / "features"
            feature_dir.mkdir(parents=True)
            torch.save(torch.zeros((2, 402)), feature_dir / "u1.pt")
            manifest = root / "data" / "eval_trials.csv"
            manifest.write_text(
                "trial_id,claimed_id,test_file\n"
                "t1,spk,u1\n"
                "t2,spk,u1\n",
                encoding="utf-8",
            )
            trials, cache = load_trials_and_features(root, "eval")
            self.assertEqual([trial.trial_id for trial in trials], ["t1", "t2"])
            self.assertEqual(list(cache), ["u1"])
            self.assertTrue(all(trial.label is None for trial in trials))

    def test_submission_validation_checks_exact_ids_and_numeric_scores(self) -> None:
        trial = Trial("eval-id", "speaker", "file")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["trial_id", "score"])
                writer.writeheader()
                writer.writerow({"trial_id": "eval-id", "score": "0.125"})
            _validate_submission(path, [trial])
            path.write_text("trial_id,score\neval-id,nan\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not finite"):
                _validate_submission(path, [trial])


if __name__ == "__main__":
    unittest.main()
