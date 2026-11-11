# Task 6 — Analysis and Ablation

All comparisons use the same 1,824 labeled development trials, with 196 genuine and 1,628 impostor trials. Evaluation trials, labels, and scores were not read.

## Relevance factor versus EER

Reused the Task 4 development results for r=4, 8, 16, and 32. The changed setting is the MAP relevance factor; the UBM, all three enrolment utterances, feature representation, and trial set are fixed.

| r | Development EER |
|---:|---:|
| r=4 | 0.295994 |
| r=8 | 0.224346 |
| r=16 | 0.153005 |
| r=32 | 0.112326 |

EER decreases over the tested range as the relevance factor increases. In this experiment, stronger shrinkage toward the shared UBM produced better development verification than allowing the limited per-speaker enrolment evidence to move means more freely. This is an observed trend only within r=4–32.

## Enrolment size versus EER

Re-adapted 97 development speakers with the unchanged 64-component diagonal UBM and r=32, selecting the first 1, 2, or 3 recordings per speaker after sorting by `enrol_index`. The changed setting is enrolment utterance count; MAP rule, UBM, features, test trials, and scoring are fixed. Every setting was scored against the same 1824 development trials.

| Enrolment utterances | Development EER |
|---:|---:|
| 1 | 0.188675 |
| 2 | 0.143296 |
| 3 | 0.112326 |

More enrolment utterances provide additional speaker evidence for the adapted means. The measured EERs show the actual effect on this development set; the three-utterance result should match Task 4's r=32 result because it uses the same recordings and pipeline.

## MAP versus independent speaker GMM

Reused the Task 4 comparison on the same development speakers, all three enrolment recordings, features, and trials. Only the speaker-model estimation method differs.

| Method | Development EER |
|---|---:|
| MAP mean-only (r=32) | 0.112326 |
| Independent diagonal GMM (64 components) | 0.479661 |

The MAP model reuses the background UBM and adapts means only, while the independent model estimates its own GMM parameters from each speaker's small enrolment set. The Task 4 results show how these approaches compare under that data constraint.

## Artifacts

- `tables/comparison.csv`: all nine comparison rows and fixed/changed-setting descriptions.
- `tables/relevance_factor.csv`, `tables/enrolment_size.csv`, and `tables/map_vs_independent_gmm.csv`: focused comparison tables.
- `plots/relevance_factor_vs_eer.svg`, `plots/enrolment_size_vs_eer.svg`, and `plots/map_vs_independent_gmm.svg`.
- Relevance-factor and model-method results were reused from `results/task4/tables/development_comparison.csv`; only the enrolment-size ablation was newly run.
