from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.build_static_dataset import BuildDatasetConfig, build_static_dataset
from src.data.static_dataset import (
    SplitConfig,
    WindowingConfig,
    assign_trajectory_splits,
    build_window_dataset_for_split,
    load_manifest,
    vet_dataset,
)


class TestDataPipeline(unittest.TestCase):
    def test_build_vet_split_and_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "dataset"
            cfg = BuildDatasetConfig(
                seeds=(0, 1),
                regimes=("healthy", "pathological"),
                t_end_ms=1200.0,
                t_warmup_ms=500.0,
                include_state=False,
                output_dir=str(out_dir),
            )
            built_dir = build_static_dataset(cfg)

            self.assertTrue((built_dir / "manifest.csv").exists())
            self.assertTrue((built_dir / "build_config.yaml").exists())

            manifest = load_manifest(built_dir)
            self.assertEqual(len(manifest), 4)

            valid_df, issues_df = vet_dataset(
                built_dir, expected_fs_hz=250.0, min_samples=128
            )
            self.assertEqual(len(valid_df), 4)
            self.assertEqual(len(issues_df), 0)

            split_df = assign_trajectory_splits(valid_df, SplitConfig())
            self.assertEqual(
                set(split_df["split"].tolist()) <= {"train", "val", "test"}, True
            )

            train_ids = set(
                split_df.loc[split_df["split"] == "train", "trajectory_id"].tolist()
            )
            val_ids = set(
                split_df.loc[split_df["split"] == "val", "trajectory_id"].tolist()
            )
            test_ids = set(
                split_df.loc[split_df["split"] == "test", "trajectory_id"].tolist()
            )
            self.assertEqual(len(train_ids & val_ids), 0)
            self.assertEqual(len(train_ids & test_ids), 0)
            self.assertEqual(len(val_ids & test_ids), 0)

            window_cfg = WindowingConfig(
                window_length=128, stride_ms=50.0, sampling_rate_hz=250.0
            )
            x_train, y_train, _ = build_window_dataset_for_split(
                dataset_dir=built_dir,
                manifest_with_split=split_df,
                split="train",
                window_cfg=window_cfg,
            )

            self.assertEqual(x_train.ndim, 2)
            self.assertEqual(y_train.ndim, 1)
            self.assertEqual(x_train.shape[1], 128)
            self.assertEqual(x_train.shape[0], y_train.shape[0])
            if x_train.shape[0] > 0:
                self.assertTrue(np.all(np.isfinite(x_train)))
                self.assertTrue(set(np.unique(y_train).tolist()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
