"""Vet static trajectories and assign train/val/test splits.

Usage:
    uv run python train/prepare-static-splits.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.static_dataset import (
    SplitConfig,
    WindowingConfig,
    assign_trajectory_splits,
    build_window_dataset_for_split,
    vet_dataset,
)


def main() -> None:
    dataset_dir = Path("artifacts/datasets/static_v1")
    valid_df, issues_df = vet_dataset(dataset_dir)

    issues_path = dataset_dir / "vet_issues.csv"
    if len(issues_df) > 0:
        issues_df.to_csv(issues_path, index=False)

    split_df = assign_trajectory_splits(valid_df, SplitConfig())
    split_manifest_path = dataset_dir / "manifest_with_splits.csv"
    split_df.to_csv(split_manifest_path, index=False)

    split_scenario_counts_path = dataset_dir / "split_scenario_counts.csv"
    split_df.groupby(["split", "scenario"]).size().reset_index(
        name="n_trajectories"
    ).to_csv(
        split_scenario_counts_path,
        index=False,
    )

    window_cfg = WindowingConfig()
    window_count_rows: list[dict[str, object]] = []
    for split_name in ["train", "val", "test", "holdout"]:
        x, y, meta = build_window_dataset_for_split(
            dataset_dir=dataset_dir,
            manifest_with_split=split_df,
            split=split_name,
            window_cfg=window_cfg,
        )
        _ = x, y
        if len(meta) == 0:
            continue
        counts = meta.groupby(["subset"]).size().reset_index(name="n_windows")
        for _, row in counts.iterrows():
            window_count_rows.append(
                {
                    "split": split_name,
                    "subset": row["subset"],
                    "n_windows": int(row["n_windows"]),
                }
            )

    window_counts_path = dataset_dir / "window_subset_counts.csv"
    if len(window_count_rows) > 0:
        pd.DataFrame(window_count_rows).to_csv(window_counts_path, index=False)

    print(f"Valid trajectories: {len(valid_df)}")
    print(f"Issue rows: {len(issues_df)}")
    if len(issues_df) > 0:
        print(f"Issues report: {issues_path}")
    print(f"Split manifest: {split_manifest_path}")
    print(f"Split scenario counts: {split_scenario_counts_path}")
    if len(window_count_rows) > 0:
        print(f"Window subset counts: {window_counts_path}")
    print(split_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
