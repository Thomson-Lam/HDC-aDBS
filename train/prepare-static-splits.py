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

from src.data.static_dataset import SplitConfig, assign_trajectory_splits, vet_dataset


def main() -> None:
    dataset_dir = Path("artifacts/datasets/static_v1")
    valid_df, issues_df = vet_dataset(dataset_dir)

    issues_path = dataset_dir / "vet_issues.csv"
    if len(issues_df) > 0:
        issues_df.to_csv(issues_path, index=False)

    split_df = assign_trajectory_splits(valid_df, SplitConfig())
    split_manifest_path = dataset_dir / "manifest_with_splits.csv"
    split_df.to_csv(split_manifest_path, index=False)

    print(f"Valid trajectories: {len(valid_df)}")
    print(f"Issue rows: {len(issues_df)}")
    if len(issues_df) > 0:
        print(f"Issues report: {issues_path}")
    print(f"Split manifest: {split_manifest_path}")
    print(split_df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
