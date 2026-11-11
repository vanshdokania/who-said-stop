"""Trial manifests and Task 1 feature loading for Task 5."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch


FEATURE_DIMENSION = 402


@dataclass(frozen=True)
class Trial:
    """A claim and test recording; development trials also carry a label."""

    trial_id: str
    claimed_id: str
    test_file: str
    label: int | None = None


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required trial manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Trial manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _load_feature(feature_root: Path, file_id: str) -> torch.Tensor:
    path = feature_root / "features" / f"{file_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Feature tensor for {file_id} is missing: {path}")
    features = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Feature file is not a tensor: {path}")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected {file_id} features shaped [frames, {FEATURE_DIMENSION}], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError(f"Feature tensor is empty: {path}")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"Feature tensor has invalid dtype or values: {path}")
    return features.to(torch.float64)


def load_trials_and_features(
    project_root: Path,
    split: str,
) -> tuple[list[Trial], dict[str, torch.Tensor]]:
    """Load trials in manifest order and cache each referenced feature once."""
    if split not in ("dev", "eval"):
        raise ValueError("split must be 'dev' or 'eval'")
    manifest_path = project_root / "data" / f"{split}_trials.csv"
    fieldnames, rows = _read_rows(manifest_path)
    required = {"trial_id", "claimed_id", "test_file"}
    if split == "dev":
        required.add("label")
    if not rows or not required.issubset(fieldnames):
        raise ValueError(
            f"{manifest_path} must contain columns {sorted(required)} and at least one row"
        )

    feature_root = project_root / "features_output" / split
    trials = []
    feature_cache: dict[str, torch.Tensor] = {}
    seen_trial_ids = set()
    for row_number, row in enumerate(rows, start=2):
        trial_id = row["trial_id"].strip()
        claimed_id = row["claimed_id"].strip()
        test_file = row["test_file"].strip()
        if not trial_id or not claimed_id or not test_file:
            raise ValueError(f"Invalid trial reference in {manifest_path}:{row_number}")
        if trial_id in seen_trial_ids:
            raise ValueError(f"Duplicate trial ID {trial_id!r} in {manifest_path}")
        seen_trial_ids.add(trial_id)

        label = None
        if split == "dev":
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid development label in {manifest_path}:{row_number}"
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"Development label must be 0 or 1 at {manifest_path}:{row_number}"
                )

        if test_file not in feature_cache:
            feature_cache[test_file] = _load_feature(feature_root, test_file)
        trials.append(Trial(trial_id, claimed_id, test_file, label))
    return trials, feature_cache
