# Task 4 — MAP Adaptation and Enrolment

## Data and method

- Loaded the existing 64-component, 402-dimensional diagonal UBM from `results/task3/models/ubm_final.pt`; no UBM retraining was performed.
- Development: 97 speakers, 291 enrolment recordings, 14091 pooled frames.
- Evaluation enrolment: 179 speakers, 537 recordings, 26095 pooled frames. No evaluation trial data or labels were read.
- For each speaker, recordings were ordered by `enrol_index`, loaded with the unchanged Task 1 feature representation, and concatenated into one frame sequence.
- UBM responsibilities were computed in the log domain. Zero-order occupancies are sums of responsibilities; first-order statistics are responsibility-weighted feature sums.
- Mean-only MAP adaptation uses `mu_hat = (F + r * mu) / (N + r)` with negligible occupancy threshold 1e-08. Components at or below the threshold retain their UBM means. UBM weights and diagonal covariances are copied unchanged.

## Relevance-factor limits

As r approaches zero, a component with nonzero occupancy approaches its enrolment-data mean `F/N`; an empty/negligible component retains the UBM mean. As r approaches infinity, every adapted mean approaches its UBM mean. Thus r controls the balance: smaller values let limited enrolment evidence move means more, while larger values shrink them more strongly toward background knowledge.

## Development trial comparison

Development models were scored only on `dev_trials.csv`. Each score is the claimed-speaker average log likelihood minus the fixed UBM average log likelihood, using the test utterance features. EER is estimated by sweeping unique score thresholds; FAR and FRR are reported at the closest operating point. Evaluation trials and labels were not used.

| Model | Relevance factor | EER | FAR at EER | FRR at EER | Threshold |
|---|---:|---:|---:|---:|---:|
| MAP mean-only | 4 | 0.295994 | 0.296069 | 0.295918 | -3.541628 |
| MAP mean-only | 8 | 0.224346 | 0.224201 | 0.224490 | -1.416667 |
| MAP mean-only | 16 | 0.153005 | 0.152948 | 0.153061 | -0.329460 |
| MAP mean-only | 32 | 0.112326 | 0.112408 | 0.112245 | 0.053206 |
| Independent diagonal GMM (64 components) | — | 0.479661 | 0.479730 | 0.479592 | -112.300176 |

Selected relevance factor: **r=32**, the lowest development EER (0.112326) among the four required values.
The best tested value is at the upper end of the requested grid; no relevance factors above 32 were tested, so selection is limited to the Task 4 candidates.

### MAP versus independently fitted speaker GMMs

The independent baseline fits a separate diagonal 64-component GMM from scratch for each development speaker, using the same pooled enrolment frames, feature representation, and diagonal covariance regularization. Its development EER was 0.479661. MAP starts from the shared background UBM and only moves means, which constrains estimates for speakers with limited enrolment data; an independent model must estimate all weights, means, and variances from those limited frames.
The independent 64-component result is a controlled same-size baseline, but each speaker has only three enrolment recordings, with 73–298 pooled frames (median 132). It has much less data per component than the UBM. Development EER—not training likelihood—determines the relevance factor.
Per-speaker baseline convergence and fit times are listed in `tables/independent_gmm_fits.csv`.

## Zero-occupancy handling and invariants

Across the development adaptations, 5920 of 24832 speaker/component occupancies were zero or below the configured threshold. Such components retained their exact UBM mean. Every saved MAP model stores the shared original UBM weights and covariances; runtime checks required exact tensor equality before saving.

## Evaluation enrolment models

Applied the selected development relevance factor r=32 to evaluation enrolment recordings only. These speaker models are saved for later use; they were not scored against evaluation trials.

## Artifacts

- Relevance-factor metrics: `tables/relevance_factor_comparison.csv`.
- Development comparison including independent baseline: `tables/development_comparison.csv`.
- Speaker/component occupancy details: `tables/speaker_statistics.csv`.
- Development trial scores: `tables/development_trial_scores.csv`.
- MAP development models: `models/development_map_r4.pt`, `development_map_r8.pt`, `development_map_r16.pt`, and `development_map_r32.pt`.
- Independent development GMMs: `models/development_independent_gmms.pt`.
- Evaluation MAP models: `models/evaluation_map_r32.pt`.
- Development EER plot: `plots/development_eer_by_relevance.svg`.
