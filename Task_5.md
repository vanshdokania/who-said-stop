
# Task 5 — Verification and Scoring

## Objective

Implement speaker verification scoring using the average per-frame log-likelihood ratio (LLR).

For every trial, compare the likelihood of the test recording under the claimed speaker's adapted model against its likelihood under the Universal Background Model (UBM).

The score is:

$$
s(X,S)=\frac{1}{T}\sum_{t=1}^{T}
\left[
\log p(x_t\mid\lambda_S)
-
\log p(x_t\mid\lambda_{\mathrm{UBM}})
\right]
$$

Where:

- \(X=(x_1,\ldots,x_T)\) represents the retained feature frames.
- \(\lambda_S\) is the adapted model of the claimed speaker.
- \(\lambda_{\mathrm{UBM}}\) is the Universal Background Model.
- \(T\) is the number of retained feature frames.

A higher score indicates stronger evidence in favour of the claimed speaker identity. Scores are not automatically calibrated probabilities.

## 1. Implement trial scoring

- Load the trained UBM from Task 3.
- Load the adapted speaker models from Task 4.
- Load the test recording features using the existing Task 1 feature representation.
- For every trial, identify the claimed speaker using `claimed_id`.
- Calculate the average per-frame log-likelihood ratio.
- Use numerically stable log-domain PyTorch operations.
- Return one continuous score per trial.

Do not convert scores into binary accept/reject decisions for submission.

## 2. Development evaluation

Use `dev_trials.csv` and the corresponding development speaker models from Task 4.

- Score every supplied development trial.
- Use the ground-truth labels to distinguish genuine and impostor trials.
- Report the total number of genuine and impostor trials.
- Generate a ROC or DET curve.
- Calculate and report Equal Error Rate (EER).

## 3. Operating threshold

Select an operating threshold based on a stated verification use case.

At the selected threshold, report:

- Threshold value.
- False Acceptance Rate (FAR).
- False Rejection Rate (FRR).

Explain the trade-off between false acceptance and false rejection.

## 4. Understand the scoring outputs

Explain the distinction between:

- **Ranking score:** The continuous LLR score indicating how strongly a trial supports the claimed identity.
- **Thresholded decision:** An accept/reject decision obtained by comparing the score against a chosen threshold.
- **Calibrated probability:** A probability estimate that requires appropriate score calibration.

Do not describe the raw LLR score as a probability.

## 5. Evaluation scoring and submission

After completing development evaluation:

- Use the selected Task 4 evaluation speaker models.
- Score every trial in `eval_trials.csv`.
- Do not use evaluation labels or tune the system using evaluation results.
- Generate a CSV submission containing continuous scores.

The submission must have exactly these columns:

```csv
trial_id,score
```

Requirements:

- One row per evaluation trial.
- Preserve the supplied trial IDs.
- Every score must be finite and numeric.
- Do not submit binary predictions.
- Follow the ordering of the supplied evaluation trial list.

Save the final file as `submission.csv`.

## 6. Expected outputs

Organize Task 5 results under:

```text
results/
    task5/
        tables/
        plots/
        logs/
        summary.md

submission.csv
```

Include:

- Development trial scores.
- Genuine and impostor trial counts.
- ROC or DET curve.
- EER result.
- Selected operating threshold, FAR and FRR.
- Final evaluation scores.
- Kaggle submission CSV.
- A summary of the scoring methodology and results.

## Completion checklist

- [ ] Average per-frame LLR scoring implemented in PyTorch.
- [ ] Development trials scored.
- [ ] Genuine and impostor trial counts reported.
- [ ] ROC or DET curve generated.
- [ ] Development EER calculated.
- [ ] Operating threshold selected and justified.
- [ ] FAR and FRR reported at the selected threshold.
- [ ] Ranking score, thresholded decision and calibrated probability explained.
- [ ] All evaluation trials scored.
- [ ] Continuous evaluation scores generated.
- [ ] Submission CSV validated.
- [ ] Final results documented.

**Stop after Task 5.**
