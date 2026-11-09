# Task 2 — GMM Behaviour Study

## Data and split

- Only Task 1 background features were loaded; development and evaluation data were not used.
- Split method: deterministic `torch.randperm` of sorted unique speaker IDs, seed 2026.
- Speakers: 629 fitting / 157 held out of 786 total.
- Recordings: 1141 fitting / 295 held out.
- Fitting observations: 10064 frames, at most 16 sampled frames per fitting speaker (seed 2027); the held-out score uses all 17339 held-out frames.
- The split is speaker-disjoint and is saved in `tables/speaker_split.csv` and `tables/speaker_split.json`.

## Covariance and component-count comparison

AIC/BIC use the fitting log likelihood and the same sampled fitting observations for every model. Lower is preferred. Held-out average log likelihood is scored on all held-out background frames; higher is preferred.

| Covariance | Components | Parameters | AIC | BIC | Held-out avg. log-likelihood | Iterations | Converged |
|---|---:|---:|---:|---:|---:|---:|:---:|
| full | 1 | 81405 | -13533798.163 | -12946321.073 | 670.506313 | 3 | yes |
| full | 2 | 162811 | -13459473.917 | -12284512.521 | 669.144750 | 4 | yes |
| full | 4 | 325623 | -13307241.432 | -10957311.422 | 663.647258 | 4 | yes |
| full | 8 | 651247 | -13172752.915 | -8472885.679 | 666.051644 | 4 | yes |
| diagonal | 1 | 804 | -13160902.363 | -13155100.120 | 654.255205 | 3 | yes |
| diagonal | 2 | 1609 | -13284330.419 | -13272718.717 | 658.830280 | 9 | yes |
| diagonal | 4 | 3219 | -13427983.787 | -13404753.166 | 663.666516 | 12 | yes |
| diagonal | 8 | 6439 | -13536461.299 | -13489992.839 | 668.058625 | 16 | yes |
| tied | 1 | 81405 | -13533798.163 | -12946321.073 | 670.506313 | 3 | yes |
| tied | 2 | 81808 | -13532883.494 | -12942498.066 | 670.500820 | 3 | yes |
| tied | 4 | 82614 | -13531824.295 | -12935622.190 | 670.491168 | 3 | yes |
| tied | 8 | 84226 | -13533752.284 | -12925916.827 | 670.554973 | 4 | yes |
| spherical | 1 | 403 | -13070496.150 | -13067587.812 | 650.205049 | 3 | yes |
| spherical | 2 | 807 | -13152525.862 | -13146701.969 | 652.958009 | 4 | yes |
| spherical | 4 | 1615 | -13229821.617 | -13218166.614 | 655.975036 | 9 | yes |
| spherical | 8 | 3231 | -13269109.439 | -13245792.217 | 657.566756 | 11 | yes |

- Lowest BIC: **diagonal / 8 components** (BIC -13489992.839).
- Highest held-out average log likelihood: **tied / 8 components** (670.554973).
- These criteria can select different models; the BIC choice is a candidate for Task 3, not an instruction to ignore held-out behaviour.

## Initialization behaviour

The repeated-start study used 20 separate random fits of a diagonal, 4-component GMM. The plotted quantity is final training log likelihood per frame.

- Best: 6718343.088624 (total; seed 31).
- Median: 6717021.569813 (total).
- Worst: 6714882.707090 (total; seed 34).
- First single run (seed 20): 6717302.453811 (total).
- Best-of-20 improvement over that single run: 1040.634813 total log-likelihood.
- Selecting the best start helps because EM can converge to different local optima from different initial parameters.

The SVG distribution plot is `plots/random_initialization_distribution.svg`; run-level values are in `tables/initialization_runs.csv`.

### Random vs. k-means initialization

| Initialization | Train avg. log-likelihood | Held-out avg. log-likelihood | Iterations | Converged |
|---|---:|---:|---:|:---:|
| kmeans | 667.521489 | 663.637297 | 5 | yes |
| random | 667.444636 | 663.623574 | 12 | yes |

By held-out average log likelihood, **kmeans** initialization was better in this run. Both methods used the same diagonal, 4-component model, fitting set, regularization, and seed.

## Degenerate component and remedy

Eight identical 3-dimensional observations produce a zero singular covariance. Direct Cholesky factorization reported info=1. Adding `epsilon I` with epsilon=1.0e-04 made Cholesky succeed (info=0); the resulting average log likelihood was finite (11.058695).

## Carry-forward observation

The BIC-preferred Task 2 candidate is diagonal covariance with 8 components. Consider it alongside the held-out ranking before choosing a UBM in Task 3. All fitting and held-out comparisons used the Task 1 402-dimensional feature representation without alteration.

## Reproducibility and files

- Covariance grid: `tables/covariance_comparison.csv` and `.json`.
- Initialization runs/comparison/summary: `tables/initialization_runs.csv`, `tables/initialization_comparison.csv`, and `tables/initialization_summary.json`.
- Degeneracy details: `tables/degeneracy_demo.json`.
- Execution output: `logs/run_all.log`.
- GMM fitting uses PyTorch only; the distribution plot is generated directly as SVG so no plotting package is required.
