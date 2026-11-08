# Task 1: Phase-Based Feature Extraction

## 1. Objective

Build a modular feature extractor that converts each speech recording
into a sequence of numerical feature vectors.

The features used as input to the speaker-verification model must be
derived from the **phase of the Fourier transform**, not from its
magnitude.

Every recording in the competition contains the same word, "stop". The
purpose of this task is to represent how the word is spoken in a way
that can later help distinguish speakers.

## 2. Required processing pipeline

The feature extractor should perform the following steps:

1.  **Load the audio**
    -   Read the recording.
    -   Check its sample rate, number of channels, length and format.
    -   Apply any required audio preparation consistently.
2.  **Divide the recording into short frames**
    -   Use short, usually overlapping frames.
    -   A suggested starting point is a 25 ms window and a 10 ms hop.
    -   These are example settings, not mandatory fixed values. Record
        the settings you choose.
3.  **Apply a window function**
    -   Apply a suitable window, such as a Hamming window, to each frame
        before the Fourier transform.
    -   Explain the choice in the report.
4.  **Calculate the short-time Fourier transform**
    -   Use `torch` or `torchaudio` to calculate the complex short-time
        Fourier transform (STFT).
    -   The complex result contains magnitude and phase information.
5.  **Create phase-derived features**
    -   Choose and implement one or more suitable phase-based
        representations.
    -   The selected features must genuinely be derived from
        Fourier-transform phase.
    -   Explain how the final feature values are calculated.
6.  **Remove or ignore silent frames**
    -   Identify frames containing little or no speech and remove or
        ignore them.
    -   Magnitude or energy may be used for silence detection only.
    -   Document the silence-detection method.
7.  **Normalise the features**
    -   Apply suitable feature normalisation.
    -   If useful, include temporal derivatives (for example, changes
        between neighbouring frames).
    -   Use a consistent procedure across all dataset splits.
8.  **Return feature tensors**
    -   Return a sequence of feature vectors for each recording.
    -   Record the final feature dimension and all relevant processing
        settings.

## 3. Permitted phase-feature choices

Possible feature representations include:

-   Direct phase features or phase spectrograms.
-   Group-delay features or group-delay cepstra.
-   Phase-difference features.
-   Temporal phase differences.
-   Baseband phase differences.
-   Relative phase shifts.
-   Instantaneous-frequency deviations.
-   Linear-prediction-residual-based phase features.
-   Learned or model-based phase encodings, such as CosPhase principal
    components.

These are possible choices; Task 1 does not require implementing every
method.

For residual-based, learned or model-based features, explain clearly how
the final representation is derived from phase. Merely naming a method
is not enough.

A suitable Mel or Bark filterbank may be applied to a phase-derived
representation. Multiple feature streams may also be combined. If using
phase angles, explain how phase wrapping is handled. Do not blindly
average raw phase angles.

## 4. Important restriction: magnitude features are not allowed as model inputs

The following must **not** be used as model-input features:

-   Magnitude spectrograms.
-   Log-Mel spectrograms based on magnitude or power.
-   Magnitude-based MFCCs.

Magnitude or signal energy may be used only for:

-   Detecting silence.
-   Numerical stabilisation.

If used for either purpose, document exactly how it is used. It must not
become part of the feature vector supplied to the GMM.

## 5. Implementation requirements

The core feature-extraction implementation must use **PyTorch (`torch`)
and `torchaudio` only**.

-   Use `torch` for tensor operations and mathematical calculations.
-   Use `torchaudio` for supported audio-processing operations and
    transforms.
-   `torch.stft` or an appropriate torchaudio transform may be used to
    obtain the complex STFT.
-   A `torchaudio.transforms.MelSpectrogram` can be used as a reference
    for how to structure a reusable transform, but its standard
    magnitude/power output is **not** a compliant phase feature.
-   Do not use NumPy, SciPy, librosa, scikit-learn, TensorFlow or other
    libraries to implement core feature extraction.
-   Python's standard library may be used for file handling, CSV
    processing, configuration and logging.
-   Plotting libraries may be used for visualisation only, not for
    feature extraction.

## 6. Use exactly the same feature pipeline for every dataset split

Apply the same feature definition and processing procedure to:

-   Background-training recordings.
-   Enrolment recordings.
-   Development/validation recordings.
-   Evaluation/trial recordings.

Keep the feature method and its processing settings consistent. For
example, do not use different frame lengths, hop lengths or feature
calculations for enrolment and trial recordings.

If a feature transformation is learned, or dataset-level normalisation
statistics are calculated, fit them using only permitted
background-training data. Do not fit them using held-out trial
recordings.

## 7. Numerical and edge-case handling

The implementation should explicitly consider:

-   **Phase wrapping:** Phase angles wrap at the boundary between `-π`
    and `+π`. Explain how the chosen representation handles this.
-   **Low-energy frequency bins:** Phase can be unreliable when a
    frequency bin has very little energy. Define how such bins are
    handled.
-   **Silent recordings or frames:** Define what happens when no speech
    frames remain after silence removal.
-   **Numerical stability:** Avoid invalid values such as NaN or
    infinity. Use suitable safeguards where needed.

These choices should be documented and tested.

## 8. Suggested modular code organisation

Keep each part of the pipeline in a small, reusable module. One possible
structure is:

``` text
who_said_stop/
├── data/                         # Original competition data; do not modify
├── src/
│   └── features/
│       ├── __init__.py
│       ├── audio.py               # Load and prepare audio
│       ├── framing.py             # Split audio into frames
│       ├── phase.py               # Calculate phase-derived features
│       ├── silence.py             # Detect and remove silent frames
│       ├── normalization.py       # Normalise features
│       └── extractor.py           # Connect the processing steps
├── scripts/
│   └── extract_features.py        # Run extraction over a dataset split
├── tests/
│   ├── test_framing.py
│   ├── test_phase.py
│   └── test_extractor.py
├── features_output/               # Saved extracted features
├── config.py                      # Shared processing settings
└── README.md
```

This is a suggested organisation, not a required exact folder structure.
Keep the original dataset files in place. Use the CSV metadata to locate
recordings and associate them with speakers or trials.

## 9. What Task 1 should produce

By the end of Task 1, the project should have:

1.  A working, modular PyTorch/torchaudio feature extractor.
2.  A clearly defined phase-derived feature representation.
3.  A consistent pipeline that can process background, enrolment,
    development and evaluation recordings.
4.  A documented method for silence removal, phase wrapping, low-energy
    bins and normalisation.
5.  The final feature dimension and all processing parameters.
6.  Tests showing that the individual processing steps and complete
    extractor behave as expected.

The output for one recording will normally be a two-dimensional tensor:

``` text
[number_of_retained_frames, feature_dimension]
```

The number of retained frames can vary between recordings. The feature
dimension should remain consistent for the chosen feature definition.

## 10. Guidance for AI coding assistants

When helping with Task 1:

-   Read this document before proposing or changing code.
-   Inspect the existing project and dataset structure before assuming
    file paths or CSV columns.
-   Do not modify, move, rename or delete original dataset files.
-   Do not choose a phase-feature method without explaining the options
    and trade-offs first.
-   Do not generate the entire implementation in one step.
-   Work module by module and explain each important operation in plain
    language.
-   Use only `torch` and `torchaudio` for core feature extraction.
-   Add small tests and run them before moving to the next module.
-   Ask for approval before making major design decisions or modifying
    project files.

The goal is to build a feature extractor that is correct, reproducible
and understood by the project developer---not merely code that runs
once.
