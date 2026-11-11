
# Task 6 — Analysis and Ablation

## Objective

Perform at least three ablation experiments to understand how different system configurations affect speaker-verification performance.

All ablations must be evaluated using development trials. Keep other settings fixed wherever possible so that the effect of each change can be understood.

## 1. Relevance factor versus EER

Study how the MAP adaptation relevance factor affects verification performance.

Use the existing Task 4 experiments:

- r = 4
- r = 8
- r = 16
- r = 32

For each value, report development EER.

Plot relevance factor against EER and briefly explain the observed trend.

Reuse existing Task 4 results wherever possible. Do not repeat experiments unnecessarily.

## 2. Enrolment size versus EER

Study the effect of the number of enrolment utterances available per speaker.

Compare:

- 1 enrolment utterance.
- 2 enrolment utterances.
- 3 enrolment utterances.

Use `enrol_index` to select the appropriate recordings for each speaker.

For every enrolment size:

1. Pool the selected enrolment features.
2. Perform MAP mean adaptation using the same UBM.
3. Keep the selected relevance factor fixed at r = 32.
4. Score the same development trials.
5. Calculate development EER.

Keep the UBM, covariance structure, feature representation and scoring method unchanged.

Report the results in a table and plot enrolment size against EER.

Briefly explain how limited enrolment data affects speaker modelling.

## 3. MAP adaptation versus independent speaker GMMs

Compare:

- MAP-adapted speaker models.
- Independently trained speaker GMMs.

Reuse the Task 4 development comparison wherever possible.

Report development EER for both approaches and briefly explain the observed difference.

Keep the development speakers, enrolment recordings, feature representation and trial set consistent.

## 4. Results and analysis

Organize Task 6 outputs under:

```text
results/
    task6/
        tables/
        plots/
        summary.md
```

Include:

- A comparison table containing all three ablation experiments.
- Relevance factor versus EER plot.
- Enrolment size versus EER plot.
- MAP versus independent GMM comparison.
- Short explanations of the observed results.

For every experiment, state which settings were changed and which were kept fixed.

## Completion checklist

- [ ] Relevance factor versus EER evaluated.
- [ ] Enrolment sizes 1, 2 and 3 compared.
- [ ] MAP adaptation versus independent GMM compared.
- [ ] All experiments evaluated on development trials.
- [ ] Other settings kept fixed wherever possible.
- [ ] Results tables generated.
- [ ] Required plots generated.
- [ ] Short analysis documented.

**Stop after Task 6. Do not introduce additional experiments beyond these three unless required by the competition rubric.**
