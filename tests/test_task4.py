import unittest

from experiments.task4.evaluation import verification_metrics


class Task4EvaluationTests(unittest.TestCase):
    def test_eer_is_zero_for_perfectly_separated_trials(self) -> None:
        rows = [
            {"label": 1, "score": 2.0},
            {"label": 1, "score": 3.0},
            {"label": 0, "score": 0.0},
            {"label": 0, "score": 1.0},
        ]
        metrics = verification_metrics(rows)
        self.assertEqual(metrics["eer"], 0.0)
        self.assertEqual(metrics["far_at_eer"], 0.0)
        self.assertEqual(metrics["frr_at_eer"], 0.0)

    def test_eer_keeps_equal_score_trials_in_one_threshold_group(self) -> None:
        rows = [
            {"label": 1, "score": 1.0},
            {"label": 0, "score": 1.0},
            {"label": 1, "score": 0.0},
            {"label": 0, "score": 0.0},
        ]
        metrics = verification_metrics(rows)
        self.assertEqual(metrics["eer"], 0.5)
        self.assertEqual(metrics["far_at_eer"], 0.5)
        self.assertEqual(metrics["frr_at_eer"], 0.5)

    def test_eer_requires_both_trial_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both target and non-target"):
            verification_metrics([{"label": 1, "score": 0.2}])


if __name__ == "__main__":
    unittest.main()
