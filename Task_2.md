# Task 2 — GMM Behaviour Study

## Goal

Study how Gaussian Mixture Models (GMMs) behave on the provided **background-speaker training data**. Complete the experiments listed below and use the results to justify the GMM configuration used later for UBM training.

**Keep this task limited to the requirements below.** Do not start Task 3 (UBM training) here.

---

## 1. Prepare the background-data split

Divide the permitted background-speaker data into:

- **Fitting subset:** used to fit GMMs and calculate AIC/BIC.
- **Held-out subset:** not used during fitting; used only to calculate average log-likelihood on unseen background data.

Split by **speaker**, not randomly by individual recordings, so that recordings from the same speaker do not appear in both subsets. Keep the split fixed and document how it was made.

Do not use development or evaluation data for these experiments.

## 2. Compare covariance structures

Fit GMMs using all four covariance structures:

- Full
- Diagonal
- Tied
- Spherical

For each structure, evaluate suitable numbers of mixture components. Calculate and report:

- AIC
- BIC
- Average log-likelihood on the held-out background subset

Use the same fitting/held-out split when comparing configurations.

### AIC and BIC

Let:

- `k` = number of free model parameters
- `n` = number of observations used for fitting
- `L_hat` = maximum likelihood of the fitted model

The criteria are:

$$
\mathrm{AIC} = 2k - 2\ln(\hat{L})
$$

$$
\mathrm{BIC} = k\ln(n) - 2\ln(\hat{L})
$$

**Lower AIC/BIC is preferred** when comparing models fitted to the same data. Report the held-out average log-likelihood separately; higher held-out log-likelihood indicates a better fit to the held-out data.

## 3. Test random initializations

For at least one selected GMM configuration:

1. Fit the model using **at least 20 different random initializations**.
2. Use one fit per seed.
3. Record the final log-likelihood for every run.
4. Plot the distribution of final log-likelihoods (for example, a histogram or box plot).
5. Report the best, worst, and typical result (such as the median).

The purpose is to show how initialization affects the result.

## 4. Compare initialization methods

Compare:

- Random initialization
- PyTorch k-means initialization

Use the same GMM configuration and data split for a fair comparison. Report the resulting log-likelihoods and briefly state which initialization performed better in your experiment.

## 5. Demonstrate the effect of choosing the best initialization

Show how selecting the best result from multiple initializations compares with using a single run. Use the results from the repeated-initialization experiment; no separate large experiment is needed.

Briefly explain why multiple initializations can help: EM may converge to different local optima depending on its starting parameters.

## 6. Demonstrate a degenerate component and a remedy

Create or identify a case where a Gaussian component becomes degenerate (for example, its covariance becomes singular or numerically unstable).

Then demonstrate a practical remedy, such as covariance regularization by adding a small positive value to the covariance diagonal:

$$
\Sigma_{\mathrm{regularized}} = \Sigma + \epsilon I
$$

Explain what went wrong and show that the remedy makes the computation stable. Choose and document the value of `epsilon`.

---

## Implementation requirements

- Implement GMM initialization, expectation-maximization (E-step and M-step), and likelihood calculations using **PyTorch**.
- Do not use NumPy, SciPy, scikit-learn, librosa, TensorFlow, or another library for the core GMM implementation.
- Plotting libraries may be used for visualizations.
- Use the existing Task 1 feature tensors. Do not re-extract the audio unless you find and document a genuine issue with those outputs.
- Do not modify the original dataset or Task 1 outputs.
- Keep the implementation reproducible: set and record random seeds.
- Handle numerical stability, including covariance regularization and log-domain calculations where needed.

## What to include in the results

Provide a compact results table covering the tested covariance structures and mixture counts. Include AIC, BIC, and held-out average log-likelihood.

Also include:

- The fitting/held-out split method and sizes.
- The repeated-initialization distribution plot and summary.
- Random-versus-k-means initialization comparison.
- The degenerate-component example and its remedy.
- A short conclusion identifying the behaviour observed and the configuration worth carrying forward to Task 3.

## Suggested project outputs

Keep the code and results organized in the existing project. For example:

```text
who_said_stop/
│
├── data/                          # Original dataset (never modify)
│
├── features_output/               # Task 1 extracted features
│   ├── background/
│   ├── dev/
│   └── eval/
│
├── src/
│   ├── features/                  # Task 1: feature extraction
│   │   ├── audio.py
│   │   └── ...
│   │
│   ├── gmm/                       # Reusable GMM implementation
│   │   ├── __init__.py
│   │   ├── initialization.py      # Random and k-means initialization
│   │   ├── gmm.py                 # GMM model and parameters
│   │   ├── em.py                  # E-step and M-step
│   │   ├── likelihood.py          # Log probability and likelihood
│   │   └── metrics.py             # AIC and BIC
│   │
│   └── utils/
│       ├── data_loader.py         # Load feature tensors and manifests
│       ├── reproducibility.py     # Random seeds
│       └── logging_utils.py       # Experiment logging
│
├── experiments/
│   └── task2/
│       ├── prepare_split.py       # Create fitting/held-out split
│       ├── covariance_study.py    # Compare covariance structures
│       ├── initialization_study.py# 20+ seeds and init comparison
│       ├── degeneracy_study.py    # Degenerate component experiment
│       └── run_all.py             # Run Task 2 experiments
│
├── results/
│   └── task2/
│       ├── tables/
│       ├── plots/
│       ├── logs/
│       └── summary.md
│
├── tests/
│   └── test_gmm.py                # GMM correctness and stability tests
│
├── Task_1.md
├── Task2_GMM_Behaviour_Guide.md
├── README.md
├── requirements.txt
└── .gitignore