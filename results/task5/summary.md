# Task 5 — Verification and Scoring

## Scoring method

For each trial, Task 1 feature frames are scored under the claimed speaker's Task 4 mean-only MAP model and the fixed Task 3 UBM. The PyTorch GMM implementation computes each mixture log likelihood in the log domain with `logsumexp`; the trial score is the speaker model's mean frame log likelihood minus the UBM's mean frame log likelihood. Higher values favour the claimed identity.
The selected Task 4 relevance factor was r=32. No UBM training or MAP adaptation was repeated.

## Development evaluation

- Trials: 1824 total; 196 genuine and 1628 impostor.
- EER: 0.112326.
- Operating threshold: **0.053206298846703248**.
- FAR at threshold: 0.112408.
- FRR at threshold: 0.112245.
- The threshold is the development ROC point closest to FAR=FRR, appropriate as a balanced-error operating point when no application-specific error costs or target operating rate are specified. Accept when score >= threshold; otherwise reject. A higher threshold typically reduces false acceptances while increasing false rejections, and vice versa.
- A ranking score is a continuous LLR used to compare trial evidence; it is not itself a probability. A threshold turns the score into an accept/reject decision. A calibrated probability requires a separate calibration procedure and representative calibration data; none was applied here.

## Evaluation and submission

- Scored all 3524 evaluation trials using the matching evaluation speaker model for each `claimed_id`. Evaluation labels/results were not used for model selection.
- Submission: `submission.csv` with exactly `trial_id,score`, manifest order preserved, and all values finite and numeric.
- These are continuous scores, not binary predictions.

## Artifacts

- `tables/development_trial_scores.csv`: one development LLR per trial, with labels.
- `tables/evaluation_trial_scores.csv`: one evaluation LLR per trial.
- `tables/development_roc.csv`: threshold, FAR, FRR, and TAR operating points.
- `tables/development_metrics.json`: EER, selected threshold, and trial counts.
- `plots/development_roc.svg`: ROC curve with the selected operating point.
- `logs/task5.log`: run and validation counts.
