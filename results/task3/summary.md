# Task 3 — UBM Training

## Data and method

- Only Task 1 background feature tensors were used. Development and evaluation recordings were not loaded.
- Reused the fixed, speaker-disjoint Task 2 split; no Task 2 outputs were modified.
- Fitting: 629 speakers, 1141 recordings, 10064 deterministic frames (maximum 16 per speaker, sampling seed 2027).
- Held out: 157 speakers, 295 recordings, all 17339 feature frames used only for evaluation.
- Initialization: random fitting-data observations as distinct component means, equal initial weights, regularized global covariance; seed = 4300 + covariance-type index * 100 + component count.
- EM: PyTorch E/M steps; maximum 40 iterations; convergence when relative change in total fitting log likelihood is at most 0.0001.
- Covariance regularization: add 0.0001 to covariance diagonals (diagonal/spherical variances are lower-bounded by the same value).
- Scoring uses chunks of 128 observations. All models use identical fitting/held-out sets.
- One fit per requested configuration; no held-out data is used for parameter updates.

## Comparison results

| Covariance | Components | Status | Time (s) | Iterations | Converged | Train log-likelihood | Held-out avg. log-likelihood | Numerically stable |
|---|---:|---|---:|---:|:---:|---:|---:|:---:|
| diagonal | 16 | success | 2.00 | 23 | yes | 6797677.421 | 669.364217 | yes |
| diagonal | 32 | success | 2.30 | 26 | yes | 6827750.743 | 670.780245 | yes |
| diagonal | 64 | success | 3.78 | 24 | yes | 6856013.560 | 671.048486 | yes |
| diagonal | 128 | success | 5.55 | 18 | yes | 6877439.449 | 670.991778 | yes |
| diagonal | 256 | success | 8.25 | 17 | yes | 6956519.343 | 670.872951 | yes |
| spherical | 16 | success | 1.21 | 11 | yes | 6649393.238 | 658.024112 | yes |
| spherical | 32 | success | 1.71 | 14 | yes | 6666419.012 | 658.623802 | yes |
| spherical | 64 | success | 1.96 | 13 | yes | 6682147.868 | 658.675005 | yes |
| spherical | 128 | success | 2.97 | 14 | yes | 6712316.436 | 658.845292 | yes |
| spherical | 256 | success | 4.70 | 14 | yes | 6756033.635 | 658.636077 | yes |
| tied | 16 | success | 1.29 | 4 | yes | 6851747.376 | 670.389919 | yes |
| tied | 32 | success | 1.43 | 4 | yes | 6856061.063 | 670.498624 | yes |
| tied | 64 | success | 0.75 | 3 | yes | 6861375.071 | 670.501037 | yes |
| tied | 128 | success | 1.08 | 4 | yes | 6875777.294 | 670.552138 | yes |
| tied | 256 | success | 2.63 | 7 | yes | 6909354.709 | 670.799842 | yes |
| full | 16 | success | 17.23 | 5 | yes | 7594222.609 | 654.160833 | yes |
| full | 32 | success | 26.69 | 4 | yes | 7336097.746 | 666.379415 | yes |
| full | 64 | success | 53.96 | 4 | yes | 7632591.929 | 669.744188 | yes |
| full | 128 | success | 96.52 | 4 | yes | 7673446.114 | 665.604052 | yes |
| full | 256 | success | 212.23 | 4 | yes | 8123050.305 | 665.961581 | yes |

## Mixture-count and covariance observations

- Compare held-out log-likelihood across component counts within each covariance family; a higher value indicates better held-out density fit.
- Increasing component count adds representational flexibility and parameter cost. Diagonal and spherical covariance structures are substantially smaller than full covariance in 402 dimensions.
- Full covariance needs `K × D × (D + 1) / 2` covariance parameters. At high K and D=402, individual component estimates are data-hungry; regularization helps numerical factorization but does not create missing statistical support.
- Training time and convergence counts are in the table; failed or unstable fits, if any, are reported as such instead of being omitted.
- The largest adjacent decrease in a recorded total training-likelihood history was 0.032809; this small non-monotonic step occurred in a regularized full-covariance fit and is recorded in the history rather than hidden.
- Higher background likelihood is only a density-fit observation; it does not prove improved speaker-verification performance, which must be evaluated in later tasks.

## Final UBM selection

Selected **diagonal covariance with 64 components** (held-out average log likelihood 671.048486; training time 3.78s; 24 iterations; converged=true).
Selection is the numerically stable, converged successful model with the highest held-out average log likelihood. This provides an empirical background-density criterion for the later MAP adaptation stage; Task 3 does not claim verification performance.

The final PyTorch checkpoint is `models/ubm_final.pt`. It stores weights, means, covariance parameters and type, feature dimension, and reproducible training metadata. 20 of 20 requested configurations completed successfully.

## Artifacts

- `tables/ubm_comparison.csv` and `.json`: configuration metrics and fit histories.
- `plots/training_log_likelihood.svg`: per-frame training likelihood across EM iterations.
- `plots/heldout_likelihood_by_components.svg`: held-out scores across mixture counts.
- `plots/training_time.svg`: wall-clock training time per fit.
- `logs/run_all.log`: experiment log.
