# Task 1: Phase-Based Feature Extraction

This implementation produces one variable-length sequence of
phase-derived feature vectors for each recording. The same `FeatureConfig`
and processing functions are used for background, enrollment, development,
and evaluation recordings.

## Pipeline

1. `src/features/audio.py` reads PCM16 WAV files with Python's standard
   `wave` module and validates mono audio at 16 kHz. This avoids the blocked
   TorchCodec native extension in the current Windows environment. PCM samples
   are converted directly into a normalized float32 Torch tensor without
   NumPy or another decoder. The feature calculations remain in PyTorch.
2. `src/features/framing.py` creates overlapping 25 ms frames at a 10 ms hop.
3. `src/features/silence.py` keeps frames whose RMS is at least -50 dBFS.
   RMS is used only to decide which frames to retain; it is not returned as
   a feature.
4. `src/features/phase.py` applies a periodic Hamming window and calculates
   an uncentered complex STFT with a 400-sample FFT.
5. Adjacent STFT frames are converted to wrap-aware temporal phase features.
   A pair is retained only when both frames are classified as speech and at
   least one frequency bin has reliable magnitude.
6. `src/features/normalization.py` L2-normalizes each retained feature vector.
7. `src/features/extractor.py` connects the steps and returns a float tensor
   with shape `[retained_frames, 402]`.

An all-silent recording, or a recording too short to form two STFT frames,
returns an empty tensor with shape `[0, 402]`.

## Feature definition and choices

For complex STFT coefficient `X[t, k]` at frame `t` and frequency bin `k`,
the phase change between adjacent frames is represented by the normalized
cross-spectrum:

```text
C[t, k] = X[t, k] * conjugate(X[t-1, k])
u[t, k] = C[t, k] / abs(C[t, k])
feature[t] = [real(u[t, :]), imag(u[t, :])]
```

`real(u)` and `imag(u)` equal cosine and sine of the phase difference.
This represents the circular phase difference without subtracting wrapped
angles or averaging raw phase. The first STFT frame has no predecessor, so
it does not create an output vector. At 400-point, one-sided STFT, there are
201 frequency bins; concatenating cosine and sine gives 402 values per
retained frame.

Direct phase angles have a discontinuity at `-pi`/`+pi`, so plain angle
subtraction can report a large change for two nearby phases. Encoding the
temporal phase difference with sine and cosine removes that boundary
discontinuity and avoids a learned transform. The trade-off is that absolute
phase is discarded and temporal differences can be sensitive to noise or
phase changes in weak bins. Group delay could represent frequency-direction
phase variation, but requires frequency-neighbor handling and was not needed
for this first, inspectable phase-only implementation.

Magnitude is used only in two permitted ways: RMS for silence detection, and
a per-frame relative threshold (`1e-4` of that frame's peak) to exclude
unreliable low-energy frequency bins. A bin contributes only when it passes
the threshold in both adjacent frames. No magnitude or energy values are
included in the feature vector.

The encoded cosine/sine pairs are bounded and describe unit-circle phase.
After unreliable bins are zeroed, each feature vector is L2-normalized across
its 402 values. This per-recording, per-frame operation does not learn
statistics from held-out data or mix magnitude into the model input. A frame
with no reliable bins is excluded; zero-energy frames are excluded by the
RMS mask. Numerical checks reject non-finite intermediate input and guarded
division prevents zero denominators.

## Parameters

`config.py` is the single source of processing settings:

| Setting | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Frame duration | 25 ms (400 samples) |
| Hop duration | 10 ms (160 samples) |
| FFT length | 400 samples |
| Window | Periodic Hamming |
| STFT centering | Disabled |
| Silence threshold | -50 dBFS frame RMS |
| Low-energy bin threshold | `1e-4` of per-frame peak magnitude |
| Normalization | Per-frame L2 |
| Feature dimension | 402 |

No resampling or channel mixing is performed: the audio loader validates
the corpus contract (mono, 16 kHz) and raises an explicit error if a file
does not match it.

## Running extraction

From the project root, run:

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py --split all
```

Use `--split background`, `--split dev`, or `--split eval` to process only
one split. Outputs are saved under `features_output/<split>/`: one `.pt`
tensor per unique audio ID, an `index.csv` connecting manifest rows to
features, and a `metadata.json` recording the settings. Repeated trial-file
references reuse a single extracted tensor while retaining every trial row
in the index. Original audio and CSV files are read-only inputs.

## Tests

Run the standard-library test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover audio validation, framing, phase wrapping, weak-bin masking,
silence handling, normalization, configuration, and the complete waveform
to feature-tensor path.

## Task 2: GMM behaviour study

The GMM implementation in `src/gmm/` uses PyTorch for initialization,
expectation-maximization, covariance calculations, and log-domain likelihoods.
Run all Task 2 experiments from the project root with:

```powershell
.\.venv\Scripts\python.exe -m experiments.task2.run_all
```

Only `features_output/background/` is loaded. A fixed 80/20 split is made by
speaker with seed 2026; 16 fitting frames per speaker are sampled
deterministically (seed 2027), while every held-out frame is used for
evaluation. The study compares 1, 2, 4, and 8 components with full, diagonal,
tied, and spherical covariance. It also runs 20 random starts of a diagonal
4-component model, compares random and k-means initialization, and tests
covariance regularization on a singular example. A standard-library SVG
histogram is written, so no plotting package is required.

Code lives under `src/gmm/` and `experiments/task2/`; generated tables, plots,
logs, and the experiment report are written under `results/task2/`.

## Task 3: UBM training

Run the Task 3 comparison with:

```powershell
.\.venv\Scripts\python.exe -m experiments.task3.run_all
```

It reuses the saved Task 2 speaker-disjoint split and deterministic
16-frames-per-fitting-speaker sample. The 16/32/64/128/256-component grid is
trained on background features only with the Task 2 PyTorch EM implementation.
Comparison tables, SVG plots, logs, the selected UBM checkpoint, and a report
are saved under `results/task3/`. The held-out background frames are scored
only after fitting and are never used for parameter updates.

## Task 4: MAP adaptation and enrolment

Run Task 4 with:

```powershell
.\.venv\Scripts\python.exe -m experiments.task4.run_all
```

The script loads the existing Task 3 diagonal 64-component UBM, pools each
speaker's enrolment features in `enrol_index` order, and performs PyTorch
mean-only MAP adaptation for relevance factors 4, 8, 16, and 32. Development
models are evaluated only with development trials; the selected factor is then
applied to evaluation enrolment features without reading evaluation trials or
labels. Adapted models, independent development GMM baselines, metrics, and
tables are written under `results/task4/`.
