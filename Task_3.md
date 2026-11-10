
# Task 3 — Train the UBM

## Objective

Train a Gaussian Mixture Model-based Universal Background Model (GMM-UBM) using pooled feature frames from the permitted background speakers.

## 1. Train the UBM

- Reuse the GMM implementation developed in Task 2.
- Train the UBM using pooled background-speaker feature frames.
- Implement the E-step and M-step using PyTorch.
- Experiment with the following mixture counts:
  - 16
  - 32
  - 64
  - 128
  - 256

## 2. Compare covariance structures

Compare the following covariance structures:

- Full
- Diagonal
- Tied
- Spherical

For each feasible configuration, record its held-out average log-likelihood, training time, convergence behaviour and numerical stability.

Justify the covariance structure selected for the final UBM.

## 3. Training configuration

Document the following:

- Initialization method.
- Maximum EM iterations.
- Convergence criterion and tolerance.
- Covariance regularization.
- Frame sampling or subsampling strategy, if used.

Use consistent frame sampling across comparable experiments.

## 4. Evaluate and compare models

For each experiment, report:

- Mixture count.
- Covariance structure.
- Training time.
- Number of iterations.
- Whether the model converged.
- Final training log-likelihood.
- Held-out average log-likelihood.

Use the held-out background speakers only for evaluation, not parameter updates.

## 5. Analyse mixture count

Explain how increasing the number of Gaussian components affects:

- Model fit.
- Numerical stability.
- Computational cost.
- Potential speaker-verification performance.

Base the discussion on experimental observations.

## 6. Select the final UBM

Choose the final configuration using:

- Held-out average log-likelihood.
- Convergence behaviour.
- Numerical stability.
- Training time and computational cost.

Explain why the selected configuration is appropriate for subsequent MAP adaptation.

Save the final UBM parameters in a PyTorch checkpoint for Task 4.

## 7. Expected outputs

Organize results under:

```text
results/
    task3/
        tables/
        plots/
        logs/
        models/
        summary.md
```

Include:

- A table comparing mixture counts and covariance structures.
- Training log-likelihood versus EM iterations plot.
- Held-out average log-likelihood versus mixture count plot.
- Training time comparison.
- Final trained UBM checkpoint.
- A summary explaining experiments, observations and final model selection.

## Completion checklist

- [ ] UBM trained with 16, 32, 64, 128 and 256 components.
- [ ] Covariance structures compared.
- [ ] E-step and M-step implemented using PyTorch.
- [ ] Initialization and convergence settings documented.
- [ ] Covariance regularization documented.
- [ ] Frame sampling strategy documented.
- [ ] Training time and convergence recorded.
- [ ] Held-out average log-likelihood calculated.
- [ ] Mixture count effects analysed.
- [ ] Final UBM configuration justified.
- [ ] Final UBM checkpoint saved.

**Stop after Task 3. Do not start MAP adaptation.**
