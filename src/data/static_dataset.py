"""Load, vet, split, and window static ODE trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    """Trajectory-level split configuration."""

    train_size: float = 0.60
    val_size: float = 0.20
    test_size: float = 0.20
    random_state: int = 123
    stratify_by_regime: bool = True


@dataclass(frozen=True)
class WindowingConfig:
    """Window extraction configuration for fixed-rate LFP."""

    window_length: int = 128
    stride_ms: float = 50.0
    sampling_rate_hz: float = 250.0

    @property
    def stride_samples(self) -> int:
        raw = self.stride_ms * self.sampling_rate_hz / 1000.0
        stride = int(round(raw))
        return max(1, stride)


def load_manifest(dataset_dir: str | Path) -> pd.DataFrame:
    """Load dataset manifest CSV."""
    path = Path(dataset_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def load_trajectory(
    dataset_dir: str | Path, relative_path: str
) -> dict[str, np.ndarray]:
    """Load one trajectory NPZ from a relative path recorded in manifest."""
    abs_path = Path(dataset_dir) / relative_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {abs_path}")
    data = np.load(abs_path)
    return {k: data[k] for k in data.files}


def _vet_single_trajectory(
    t: np.ndarray,
    lfp: np.ndarray,
    expected_fs_hz: float,
    min_samples: int,
) -> list[str]:
    issues: list[str] = []

    if t.ndim != 1:
        issues.append("t is not rank-1")
    if lfp.ndim != 1:
        issues.append("lfp is not rank-1")
    if len(t) != len(lfp):
        issues.append("t and lfp length mismatch")
    if len(lfp) < min_samples:
        issues.append(f"lfp has too few samples ({len(lfp)} < {min_samples})")

    if not np.all(np.isfinite(t)):
        issues.append("t has non-finite values")
    if not np.all(np.isfinite(lfp)):
        issues.append("lfp has non-finite values")

    if len(t) > 1:
        dt = np.diff(t)
        if not np.all(dt > 0):
            issues.append("timestamps are not strictly increasing")
        expected_dt = 1000.0 / expected_fs_hz
        if np.max(np.abs(dt - expected_dt)) > 1e-3:
            issues.append("sample interval deviates from expected fixed-rate grid")

    return issues


def vet_dataset(
    dataset_dir: str | Path,
    expected_fs_hz: float = 250.0,
    min_samples: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run basic quality checks over all trajectories.

    Returns:
        (valid_rows, issues_df)
    """
    manifest = load_manifest(dataset_dir)
    issue_rows: list[dict[str, str]] = []
    valid_indices: list[int] = []

    for idx, row in manifest.iterrows():
        traj = load_trajectory(dataset_dir, row["path"])
        issues = _vet_single_trajectory(
            t=np.asarray(traj["t"]),
            lfp=np.asarray(traj["lfp"]),
            expected_fs_hz=expected_fs_hz,
            min_samples=min_samples,
        )
        if len(issues) == 0:
            valid_indices.append(idx)
        else:
            issue_rows.append(
                {
                    "trajectory_id": str(row["trajectory_id"]),
                    "issues": "; ".join(issues),
                }
            )

    valid_df = manifest.iloc[valid_indices].reset_index(drop=True)
    issues_df = pd.DataFrame(issue_rows)
    return valid_df, issues_df


def assign_trajectory_splits(
    manifest: pd.DataFrame,
    split_cfg: SplitConfig,
) -> pd.DataFrame:
    """Assign train/val/test splits at trajectory level.

    Uses sklearn train_test_split and optional regime stratification.
    """
    total = split_cfg.train_size + split_cfg.val_size + split_cfg.test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    if len(manifest) < 3:
        raise ValueError("Need at least 3 trajectories for train/val/test split")

    df = manifest.copy().reset_index(drop=True)

    strat = df["regime"] if split_cfg.stratify_by_regime else None
    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=split_cfg.train_size,
            random_state=split_cfg.random_state,
            stratify=strat,
        )
    except ValueError:
        train_df, temp_df = train_test_split(
            df,
            train_size=split_cfg.train_size,
            random_state=split_cfg.random_state,
            stratify=None,
        )

    rel_val = split_cfg.val_size / (split_cfg.val_size + split_cfg.test_size)
    temp_strat = temp_df["regime"] if split_cfg.stratify_by_regime else None
    try:
        val_df, test_df = train_test_split(
            temp_df,
            train_size=rel_val,
            random_state=split_cfg.random_state,
            stratify=temp_strat,
        )
    except ValueError:
        val_df, test_df = train_test_split(
            temp_df,
            train_size=rel_val,
            random_state=split_cfg.random_state,
            stratify=None,
        )

    train_ids = set(train_df["trajectory_id"].tolist())
    val_ids = set(val_df["trajectory_id"].tolist())
    test_ids = set(test_df["trajectory_id"].tolist())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Split leakage: trajectory IDs overlap across splits")

    out = df.copy()
    out["split"] = ""
    out.loc[out["trajectory_id"].isin(train_ids), "split"] = "train"
    out.loc[out["trajectory_id"].isin(val_ids), "split"] = "val"
    out.loc[out["trajectory_id"].isin(test_ids), "split"] = "test"
    return out


def extract_windows(
    lfp: np.ndarray,
    window_cfg: WindowingConfig,
) -> np.ndarray:
    """Extract fixed-length rolling windows from one LFP trajectory."""
    x = np.asarray(lfp, dtype=np.float64)
    w = window_cfg.window_length
    s = window_cfg.stride_samples
    if x.ndim != 1:
        raise ValueError("lfp must be rank-1")
    if len(x) < w:
        return np.empty((0, w), dtype=np.float64)

    starts = np.arange(0, len(x) - w + 1, s)
    windows = np.empty((len(starts), w), dtype=np.float64)
    for i, start in enumerate(starts):
        windows[i] = x[start : start + w]
    return windows


def build_window_dataset_for_split(
    dataset_dir: str | Path,
    manifest_with_split: pd.DataFrame,
    split: str,
    window_cfg: WindowingConfig,
    label_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load trajectories for one split and return (windows, labels, rows_used)."""
    if label_map is None:
        label_map = {"healthy": 0, "pathological": 1}

    rows = manifest_with_split[manifest_with_split["split"] == split].copy()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    kept_rows: list[int] = []

    for idx, row in rows.iterrows():
        regime = str(row["regime"])
        if regime not in label_map:
            continue
        traj = load_trajectory(dataset_dir, row["path"])
        windows = extract_windows(np.asarray(traj["lfp"]), window_cfg=window_cfg)
        if windows.shape[0] == 0:
            continue
        label = label_map[regime]
        xs.append(windows)
        ys.append(np.full((windows.shape[0],), label, dtype=np.int64))
        kept_rows.append(idx)

    if len(xs) == 0:
        empty_x = np.empty((0, window_cfg.window_length), dtype=np.float64)
        empty_y = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y, rows.iloc[[]]

    x_out = np.vstack(xs)
    y_out = np.concatenate(ys)
    rows_used = rows.loc[kept_rows].reset_index(drop=True)
    return x_out, y_out, rows_used
