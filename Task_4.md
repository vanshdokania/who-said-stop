
# Task 4 — MAP Adaptation and Enrolment

## Objective

Implement Maximum A Posteriori (MAP) adaptation of UBM component means using PyTorch.

For each enrolled speaker, adapt the pretrained UBM using their permitted enrolment recordings. The required baseline is **mean-only adaptation**: retain the UBM mixture weights and covariance parameters.

## 1. Prepare enrolment data

- Use `dev_enrol.csv` and `eval_enrol.csv` for their respective speaker sets.
- Load the corresponding Task 1 feature tensors.
- Pool the feature frames from all permitted enrolment recordings belonging to each speaker.
- Use `enrol_index` to identify and select the permitted enrolment recordings when required by the experiments.
- Keep development and evaluation enrolment data separate.

## 2. Implement MAP mean adaptation

For each enrolled speaker:

1. Load the trained UBM from Task 3 (`models/ubm_final.pt`).
2. Calculate component responsibilities for every enrolment feature frame using the fixed UBM.
3. Accumulate component occupancies (zero-order statistics).
4. Accumulate component first-order statistics.
5. Calculate the adapted component means using the MAP relevance-factor formula.

For component \(k\), define:

- \(N_k\): component occupancy.
- \(F_k\): first-order statistics.
- \(\mu_k\): original UBM mean.
- \(r\): relevance factor.

The adapted mean is:

$$
\hat{\mu}_k =
\frac{F_k+r\mu_k}{N_k+r}
$$

Keep the UBM mixture weights and covariance parameters unchanged.

Implement the calculations using PyTorch.

## 3. Study the relevance factor

Experiment with:

$$
r \in \{4,8,16,32\}
$$

Compare the adapted speaker models using development data.

Explain the effect of changing the relevance factor:

- Smaller \(r\): enrolment data has greater influence on adapted means.
- Larger \(r\): adapted means remain closer to the original UBM means.

Also explain the limiting behaviour.

### As \(r \rightarrow 0\)

For components with nonzero occupancy, the adapted mean approaches the enrolment-data component mean:

$$
\hat{\mu}_k \rightarrow \frac{F_k}{N_k}
$$

For components with zero occupancy, retain the original UBM mean.

### As \(r \rightarrow \infty\)

The adapted means approach the original UBM means:

$$
\hat{\mu}_k \rightarrow \mu_k
$$

Explain why the relevance factor controls the balance between general background knowledge and speaker-specific enrolment evidence.

## 4. Handle zero-occupancy components

Some Gaussian components may receive no meaningful responsibility from a speaker's limited enrolment data.

- Detect components with zero or numerically negligible occupancy.
- Retain their original UBM means.
- Ensure that adaptation does not produce NaN or infinite values.

Document how these components are handled.

## 5. Compare MAP adaptation with independent speaker GMMs

Train separate GMMs from scratch using the enrolment recordings of each development speaker.

Compare this approach against MAP mean adaptation using the same development speakers and permitted enrolment recordings.

Explain:

- The effect of having limited enrolment data.
- Why MAP adaptation can benefit from the pretrained UBM.
- The differences in stability and model estimation.
- The observed development performance.

Use the same feature representation and a fair comparison wherever possible.

## 6. Evaluate using development data

Use development enrolment recordings to construct speaker models and `dev_trials.csv` to evaluate them.

Compare the relevance-factor settings and independent speaker GMM baseline.

Use appropriate speaker-verification metrics from Task 5, including EER, FAR and FRR where available.

Do not use evaluation trial data or labels to select the relevance factor or tune the model.

## 7. Evaluation enrolment models

After selecting the relevance factor using development results:

- Apply the selected MAP adaptation configuration to `eval_enrol.csv`.
- Generate adapted speaker models for the evaluation speakers.
- Save the adapted parameters for use in Task 5.

Do not use evaluation trial outcomes for model selection.

## 8. Expected outputs

Organize Task 4 results under:

```text
results/
    task4/
        tables/
        plots/
        models/
        summary.md
```

Include:

- Relevance-factor comparison results.
- MAP versus independently trained speaker GMM comparison.
- Development evaluation metrics.
- Zero-occupancy component handling results.
- Adapted development and evaluation speaker models.
- A summary explaining the observations and selected relevance factor.

## Completion checklist

- [ ] Enrolment features pooled separately for each speaker.
- [ ] UBM loaded from Task 3.
- [ ] Component responsibilities calculated.
- [ ] Zero-order and first-order statistics accumulated.
- [ ] MAP mean adaptation implemented in PyTorch.
- [ ] UBM weights and covariances retained.
- [ ] Relevance factors 4, 8, 16 and 32 compared.
- [ ] Limiting behaviour as \(r\) approaches zero and infinity explained.
- [ ] Zero-occupancy components handled correctly.
- [ ] MAP adaptation compared with independent speaker GMMs.
- [ ] Development results documented.
- [ ] Evaluation speaker models generated using the selected configuration.

**Stop after Task 4. Do not implement final trial scoring or submission generation yet.**
