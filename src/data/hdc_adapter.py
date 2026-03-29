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
    filter_windows_by_subset,
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
    required_scenarios = {
        "clean_healthy",
        "clean_pathological",
        "moderate",
        "onset",
        "recovery",
        "healthy_holdout",
    }

    should_rebuild = False

    if not manifest_path.exists():
        should_rebuild = True
    else:
        manifest = pd.read_csv(manifest_path)
        if "scenario" not in manifest.columns:
            should_rebuild = True
        else:
            present = set(manifest["scenario"].astype(str).unique().tolist())
            if not required_scenarios.issubset(present):
                should_rebuild = True

    if should_rebuild:
        cfg = BuildDatasetConfig(output_dir=str(dataset_path))
        build_static_dataset(cfg)
        split_path = dataset_path / "manifest_with_splits.csv"
        if split_path.exists():
            split_path.unlink()

    _read_or_create_split_manifest(dataset_path)
    return dataset_path


def load_validation_data_from_static(
    dataset_dir: str | Path,
    window_cfg: WindowingConfig | None = None,
) -> ValidationData:
    """Load static trajectories and return ValidationData for HDC search/training."""
    dataset_path = ensure_static_dataset_ready(dataset_dir)
    window_cfg = window_cfg or WindowingConfig()

    split_df = pd.read_csv(dataset_path / "manifest_with_splits.csv")

    x_train_all, y_train_all, train_meta = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="train",
        window_cfg=window_cfg,
    )
    x_val_all, y_val_all, val_meta = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="val",
        window_cfg=window_cfg,
    )

    x_train, y_train, _ = filter_windows_by_subset(
        x_train_all,
        y_train_all,
        train_meta,
        allowed_subsets={"clean"},
    )
    x_val_clean, y_val_clean, _ = filter_windows_by_subset(
        x_val_all,
        y_val_all,
        val_meta,
        allowed_subsets={"clean"},
    )
    x_val_onset, y_val_onset, _ = filter_windows_by_subset(
        x_val_all,
        y_val_all,
        val_meta,
        allowed_subsets={"onset"},
    )
    x_val_recovery, y_val_recovery, _ = filter_windows_by_subset(
        x_val_all,
        y_val_all,
        val_meta,
        allowed_subsets={"recovery"},
    )
    x_val_moderate_only, y_val_moderate_only, _ = filter_windows_by_subset(
        x_val_all,
        y_val_all,
        val_meta,
        allowed_subsets={"moderate"},
    )
    x_val_clean_for_mod, y_val_clean_for_mod, _ = filter_windows_by_subset(
        x_val_all,
        y_val_all,
        val_meta,
        allowed_subsets={"clean"},
    )

    # Moderate validation set: clean healthy/pathological context + moderate windows.
    if x_val_moderate_only.shape[0] > 0 and x_val_clean_for_mod.shape[0] > 0:
        x_val_moderate = np.vstack([x_val_clean_for_mod, x_val_moderate_only])
        y_val_moderate = np.concatenate([y_val_clean_for_mod, y_val_moderate_only])
    else:
        x_val_moderate = x_val_moderate_only
        y_val_moderate = y_val_moderate_only

    x_holdout_all, y_holdout_all, holdout_meta = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="holdout",
        window_cfg=window_cfg,
    )
    x_holdout_healthy, y_holdout_healthy, _ = filter_windows_by_subset(
        x_holdout_all,
        y_holdout_all,
        holdout_meta,
        allowed_subsets={"healthy_holdout"},
    )

    val_subset_counts = (
        val_meta["subset"].value_counts().to_dict() if len(val_meta) > 0 else {}
    )
    val_scenario_counts = (
        split_df[split_df["split"] == "val"]["scenario"].value_counts().to_dict()
        if "scenario" in split_df.columns
        else {}
    )

    if x_train.shape[0] == 0 or x_val_clean.shape[0] == 0:
        raise RuntimeError(
            "Static dataset produced empty train/clean-val windows. "
            "Check trajectory durations and split manifest."
        )
    if x_val_onset.shape[0] == 0:
        raise RuntimeError(
            "Validation onset subset is empty; robust subset generation failed. "
            f"val_scenarios={val_scenario_counts}, val_subsets={val_subset_counts}"
        )
    if x_val_recovery.shape[0] == 0:
        raise RuntimeError(
            "Validation recovery subset is empty; robust subset generation failed. "
            f"val_scenarios={val_scenario_counts}, val_subsets={val_subset_counts}"
        )
    if x_val_moderate.shape[0] == 0:
        raise RuntimeError(
            "Validation moderate subset is empty; robust subset generation failed. "
            f"val_scenarios={val_scenario_counts}, val_subsets={val_subset_counts}"
        )
    if x_holdout_healthy.shape[0] == 0:
        holdout_subset_counts = (
            holdout_meta["subset"].value_counts().to_dict()
            if len(holdout_meta) > 0
            else {}
        )
        raise RuntimeError(
            "Healthy holdout subset is empty; robust subset generation failed. "
            f"holdout_subsets={holdout_subset_counts}"
        )

    return ValidationData(
        train=SplitData(x=np.asarray(x_train), y=np.asarray(y_train)),
        val_clean=SplitData(x=np.asarray(x_val_clean), y=np.asarray(y_val_clean)),
        val_onset=SplitData(x=np.asarray(x_val_onset), y=np.asarray(y_val_onset)),
        val_recovery=SplitData(
            x=np.asarray(x_val_recovery),
            y=np.asarray(y_val_recovery),
        ),
        val_moderate=SplitData(
            x=np.asarray(x_val_moderate),
            y=np.asarray(y_val_moderate),
        ),
        holdout_healthy=SplitData(
            x=np.asarray(x_holdout_healthy),
            y=np.asarray(y_holdout_healthy),
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

    x_train_all, y_train_all, train_meta = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="train",
        window_cfg=window_cfg,
    )
    x_test_all, y_test_all, test_meta = build_window_dataset_for_split(
        dataset_dir=dataset_path,
        manifest_with_split=split_df,
        split="test",
        window_cfg=window_cfg,
    )
    x_train, y_train, _ = filter_windows_by_subset(
        x_train_all,
        y_train_all,
        train_meta,
        allowed_subsets={"clean"},
    )
    x_test, y_test, _ = filter_windows_by_subset(
        x_test_all,
        y_test_all,
        test_meta,
        allowed_subsets={"clean"},
    )
    return x_train, y_train, x_test, y_test
