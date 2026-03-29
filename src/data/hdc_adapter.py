"""Adapters from static trajectory datasets to HDC validator/trainer inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hdc.search.validator import SplitData, ValidationData

from .build_static_dataset import BuildDatasetConfig, build_static_dataset
from .static_dataset import (
    SplitConfig,
    WindowingConfig,
    assign_trajectory_splits,
    build_window_dataset_for_split,
    vet_dataset,
)


def _read_or_create_split_manifest(dataset_dir: Path) -> pd.DataFrame:
    split_path = dataset_dir / "manifest_with_splits.csv"
    if split_path.exists():
        return pd.read_csv(split_path)

    valid_df, issues_df = vet_dataset(dataset_dir)
    if len(issues_df) > 0:
        issues_path = dataset_dir / "vet_issues.csv"
        issues_df.to_csv(issues_path, index=False)

    split_df = assign_trajectory_splits(valid_df, SplitConfig())
    split_df.to_csv(split_path, index=False)
    return split_df


def ensure_static_dataset_ready(dataset_dir: str | Path) -> Path:
    """Ensure static dataset and split manifest exist.

    If manifest is missing, builds the dataset. If split manifest is missing,
    performs vetting and split assignment.
    """
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "manifest.csv"

    if not manifest_path.exists():
        cfg = BuildDatasetConfig(output_dir=str(dataset_path))
        build_static_dataset(cfg)

    _read_or_create_split_manifest(dataset_path)
    return dataset_path


def load_validation_data_from_static(
    dataset_dir: str | Path,
    window_cfg: WindowingConfig | None = None,
) -> ValidationData:
    """Load static trajectories and return ValidationData for HDC search/training.

    Current MVP maps:
    - train split -> train
    - val split -> val_clean
    - val_onset/val_recovery -> val subset columns if present, else val_clean
    """
    dataset_path = ensure_static_dataset_ready(dataset_dir)
    window_cfg = window_cfg or WindowingConfig()

    split_df = pd.read_csv(dataset_path / "manifest_with_splits.csv")

    x_train, y_train, _ = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="train",
        window_cfg=window_cfg,
    )
    x_val_clean, y_val_clean, val_rows = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="val",
        window_cfg=window_cfg,
    )

    if x_train.shape[0] == 0 or x_val_clean.shape[0] == 0:
        raise RuntimeError(
            "Static dataset produced empty train/val windows. "
            "Check trajectory durations and split manifest."
        )

    # Transitional subsets are optional at this stage. If a subset column exists,
    # use it; otherwise fallback to clean validation windows.
    if "subset" in val_rows.columns:
        onset_rows = val_rows[val_rows["subset"] == "onset"]
        recovery_rows = val_rows[val_rows["subset"] == "recovery"]

        if len(onset_rows) > 0:
            onset_df = split_df[
                split_df["trajectory_id"].isin(onset_rows["trajectory_id"])
            ]
            x_val_onset, y_val_onset, _ = build_window_dataset_for_split(
                dataset_dir=dataset_path,
                manifest_with_split=onset_df.assign(split="val"),
                split="val",
                window_cfg=window_cfg,
            )
        else:
            x_val_onset, y_val_onset = x_val_clean, y_val_clean

        if len(recovery_rows) > 0:
            recovery_df = split_df[
                split_df["trajectory_id"].isin(recovery_rows["trajectory_id"])
            ]
            x_val_recovery, y_val_recovery, _ = build_window_dataset_for_split(
                dataset_dir=dataset_path,
                manifest_with_split=recovery_df.assign(split="val"),
                split="val",
                window_cfg=window_cfg,
            )
        else:
            x_val_recovery, y_val_recovery = x_val_clean, y_val_clean
    else:
        x_val_onset, y_val_onset = x_val_clean, y_val_clean
        x_val_recovery, y_val_recovery = x_val_clean, y_val_clean

    return ValidationData(
        train=SplitData(x=np.asarray(x_train), y=np.asarray(y_train)),
        val_clean=SplitData(x=np.asarray(x_val_clean), y=np.asarray(y_val_clean)),
        val_onset=SplitData(x=np.asarray(x_val_onset), y=np.asarray(y_val_onset)),
        val_recovery=SplitData(
            x=np.asarray(x_val_recovery),
            y=np.asarray(y_val_recovery),
        ),
    )


def load_train_and_test_windows(
    dataset_dir: str | Path,
    window_cfg: WindowingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train and test window datasets from static split manifest."""
    dataset_path = ensure_static_dataset_ready(dataset_dir)
    window_cfg = window_cfg or WindowingConfig()
    split_df = pd.read_csv(dataset_path / "manifest_with_splits.csv")

    x_train, y_train, _ = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="train",
        window_cfg=window_cfg,
    )
    x_test, y_test, _ = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="test",
        window_cfg=window_cfg,
    )
    return x_train, y_train, x_test, y_test
