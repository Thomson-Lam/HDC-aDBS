"""Calibration and overfitting ablation runner for HDC training.

Phases covered:
1) threshold calibration on validation,
2) train-vs-validation overfitting diagnostics,
3) encoder normalization ablations,
4) linear readout regularization sweep.

Usage:
    uv run python train/calibrate-overfitting.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdc.encoder import EncoderConfig
from hdc.training import LinearHDCTrainer, PrototypeHDCTrainer
from src.data.hdc_adapter import (
    load_train_and_test_windows,
    load_validation_data_from_static,
)
from src.data.static_dataset import WindowingConfig


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _threshold_stats(
    scores: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    y = y.astype(np.int64)
    healthy = y == 0
    path = y == 1
    fpr = float(np.mean(pred[healthy] == 1)) if np.any(healthy) else float("nan")
    tpr = float(np.mean(pred[path] == 1)) if np.any(path) else float("nan")
    bal = float(balanced_accuracy_score(y, pred))
    return {"balanced_accuracy": bal, "fpr": fpr, "tpr": tpr}


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(scores)
    return np.concatenate(
        (
            np.array([float("-inf")], dtype=np.float64),
            uniq,
            np.array([float("inf")], dtype=np.float64),
        )
    )


def _calibrate_best_balanced(
    scores: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, float]]:
    best_t = 0.0
    best = {
        "balanced_accuracy": float("-inf"),
        "fpr": float("nan"),
        "tpr": float("nan"),
    }
    for t in _candidate_thresholds(scores):
        s = _threshold_stats(scores, y, float(t))
        if s["balanced_accuracy"] > best["balanced_accuracy"]:
            best_t = float(t)
            best = s
    return best_t, best


def _calibrate_fpr_constrained(
    scores: np.ndarray,
    y: np.ndarray,
    target_fpr: float,
) -> tuple[float, dict[str, float]]:
    best_t = float("nan")
    best = {
        "balanced_accuracy": float("-inf"),
        "fpr": float("nan"),
        "tpr": float("nan"),
    }
    feasible = False
    for t in _candidate_thresholds(scores):
        s = _threshold_stats(scores, y, float(t))
        if not np.isnan(s["fpr"]) and s["fpr"] <= target_fpr:
            feasible = True
            if s["balanced_accuracy"] > best["balanced_accuracy"]:
                best_t = float(t)
                best = s

    if feasible:
        return best_t, best

    # Fallback: choose threshold with minimum FPR, then highest balanced accuracy.
    ranked: list[tuple[float, float, float, dict[str, float]]] = []
    for t in _candidate_thresholds(scores):
        s = _threshold_stats(scores, y, float(t))
        ranked.append((s["fpr"], -s["balanced_accuracy"], float(t), s))
    ranked.sort(key=lambda x: (x[0], x[1]))
    _, _, best_t, best = ranked[0]
    return float(best_t), best


def _dataset_stats(x_train: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(x_train))
    std = float(np.std(x_train))
    if std <= 0.0:
        std = 1.0
    return mean, std


def _build_encoder_config(
    *,
    normalization: str,
    seed: int,
    x_train: np.ndarray,
    dimension: int,
    n_bins: int,
    clip_z: float,
) -> EncoderConfig:
    if normalization == "dataset_zscore":
        mean, std = _dataset_stats(x_train)
        return EncoderConfig(
            dimension=dimension,
            n_bins=n_bins,
            window_length=128,
            value_init="rff",
            clip_z=clip_z,
            seed=seed,
            normalization=normalization,
            dataset_mean=mean,
            dataset_std=std,
        )
    return EncoderConfig(
        dimension=dimension,
        n_bins=n_bins,
        window_length=128,
        value_init="rff",
        clip_z=clip_z,
        seed=seed,
        normalization=normalization,
    )


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    c_values = [float(v) for v in args.linear_c.split(",") if v.strip()]
    normalizations = [s.strip() for s in args.normalizations.split(",") if s.strip()]

    window_cfg = WindowingConfig(
        window_length=128, stride_ms=50.0, sampling_rate_hz=250.0
    )
    data = load_validation_data_from_static(args.dataset_dir, window_cfg=window_cfg)
    _, _, x_test, y_test = load_train_and_test_windows(
        args.dataset_dir, window_cfg=window_cfg
    )

    rows: list[dict[str, object]] = []

    for norm in normalizations:
        for seed in seeds:
            # Prototype sweep
            enc_cfg = _build_encoder_config(
                normalization=norm,
                seed=seed,
                x_train=data.train.x,
                dimension=args.dimension,
                n_bins=args.n_bins,
                clip_z=args.clip_z,
            )
            proto = PrototypeHDCTrainer(encoder_config=enc_cfg).fit(
                data.train.x, data.train.y
            )
            _append_rows(
                rows=rows,
                trainer=proto,
                readout_type="prototype",
                normalization=norm,
                encoder_seed=seed,
                linear_c=float("nan"),
                data=data,
                x_test=x_test,
                y_test=y_test,
                target_fpr=args.target_val_fpr,
            )

            # Linear sweep (regularization C)
            for c in c_values:
                enc_cfg = _build_encoder_config(
                    normalization=norm,
                    seed=seed,
                    x_train=data.train.x,
                    dimension=args.dimension,
                    n_bins=args.n_bins,
                    clip_z=args.clip_z,
                )
                linear = LinearHDCTrainer(
                    encoder_config=enc_cfg,
                    seed=seed,
                    c=c,
                ).fit(data.train.x, data.train.y)
                _append_rows(
                    rows=rows,
                    trainer=linear,
                    readout_type="linear",
                    normalization=norm,
                    encoder_seed=seed,
                    linear_c=c,
                    data=data,
                    x_test=x_test,
                    y_test=y_test,
                    target_fpr=args.target_val_fpr,
                )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -float(r["val_clean_balanced_accuracy"]),
            float(r["holdout_fpr"]),
        ),
    )

    csv_path = out_dir / "phase1_4_experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({k for row in rows_sorted for k in row.keys()})
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    meaningful_rows = [
        r
        for r in rows
        if np.isfinite(float(r["threshold"])) and float(r["val_clean_auroc"]) > 0.5
    ]

    best_by_holdout = sorted(
        meaningful_rows,
        key=lambda r: (
            float(r["holdout_fpr"]),
            -float(r["val_clean_balanced_accuracy"]),
        ),
    )[:10]
    best_by_balanced = sorted(
        meaningful_rows,
        key=lambda r: (
            -float(r["val_clean_balanced_accuracy"]),
            float(r["holdout_fpr"]),
        ),
    )[:10]

    summary = {
        "dataset_dir": args.dataset_dir,
        "windowing": asdict(window_cfg),
        "search_space": {
            "normalizations": normalizations,
            "seeds": seeds,
            "linear_c": c_values,
            "dimension": args.dimension,
            "n_bins": args.n_bins,
            "clip_z": args.clip_z,
            "target_val_fpr": args.target_val_fpr,
        },
        "n_rows": len(rows),
        "best_by_holdout_fpr": best_by_holdout,
        "best_by_val_balanced_accuracy": best_by_balanced,
        "artifacts": {
            "experiments_csv": str(csv_path),
        },
    }
    (out_dir / "phase1_4_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
    )
    return summary


def _append_rows(
    *,
    rows: list[dict[str, object]],
    trainer: PrototypeHDCTrainer | LinearHDCTrainer,
    readout_type: str,
    normalization: str,
    encoder_seed: int,
    linear_c: float,
    data,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_fpr: float,
) -> None:
    s_train = trainer.decision_function(data.train.x)
    s_clean = trainer.decision_function(data.val_clean.x)
    s_onset = trainer.decision_function(data.val_onset.x)
    s_recovery = trainer.decision_function(data.val_recovery.x)
    s_moderate = trainer.decision_function(data.val_moderate.x)
    s_holdout = trainer.decision_function(data.holdout_healthy.x)
    s_test = (
        trainer.decision_function(x_test) if x_test.shape[0] > 0 else np.empty((0,))
    )

    for policy_name, calibrator in (
        ("best_balanced", lambda: _calibrate_best_balanced(s_clean, data.val_clean.y)),
        (
            "fpr_constrained",
            lambda: _calibrate_fpr_constrained(
                s_clean, data.val_clean.y, target_fpr=target_fpr
            ),
        ),
    ):
        threshold, cal_stats = calibrator()
        train_stats = _threshold_stats(s_train, data.train.y, threshold)
        clean_stats = _threshold_stats(s_clean, data.val_clean.y, threshold)
        onset_stats = _threshold_stats(s_onset, data.val_onset.y, threshold)
        recovery_stats = _threshold_stats(s_recovery, data.val_recovery.y, threshold)
        moderate_stats = _threshold_stats(s_moderate, data.val_moderate.y, threshold)
        test_stats = (
            _threshold_stats(s_test, y_test, threshold)
            if x_test.shape[0] > 0 and y_test.shape[0] > 0
            else {
                "balanced_accuracy": float("nan"),
                "fpr": float("nan"),
                "tpr": float("nan"),
            }
        )

        holdout_pred = (s_holdout >= threshold).astype(np.int64)
        holdout_fpr = (
            float(np.mean(holdout_pred == 1)) if holdout_pred.size else float("nan")
        )

        row = {
            "readout": readout_type,
            "normalization": normalization,
            "encoder_seed": encoder_seed,
            "linear_c": linear_c,
            "calibration_policy": policy_name,
            "threshold": threshold,
            "calibration_val_balanced_accuracy": cal_stats["balanced_accuracy"],
            "calibration_val_fpr": cal_stats["fpr"],
            "calibration_val_tpr": cal_stats["tpr"],
            "train_balanced_accuracy": train_stats["balanced_accuracy"],
            "train_auroc": _safe_auroc(data.train.y, s_train),
            "val_clean_balanced_accuracy": clean_stats["balanced_accuracy"],
            "val_clean_auroc": _safe_auroc(data.val_clean.y, s_clean),
            "val_onset_balanced_accuracy": onset_stats["balanced_accuracy"],
            "val_onset_auroc": _safe_auroc(data.val_onset.y, s_onset),
            "val_recovery_balanced_accuracy": recovery_stats["balanced_accuracy"],
            "val_recovery_auroc": _safe_auroc(data.val_recovery.y, s_recovery),
            "val_moderate_balanced_accuracy": moderate_stats["balanced_accuracy"],
            "val_moderate_auroc": _safe_auroc(data.val_moderate.y, s_moderate),
            "test_clean_balanced_accuracy": test_stats["balanced_accuracy"],
            "test_clean_auroc": _safe_auroc(y_test, s_test)
            if s_test.size
            else float("nan"),
            "holdout_fpr": holdout_fpr,
            "generalization_gap_bal_acc": train_stats["balanced_accuracy"]
            - clean_stats["balanced_accuracy"],
            "generalization_gap_auroc": _safe_auroc(data.train.y, s_train)
            - _safe_auroc(data.val_clean.y, s_clean),
        }
        rows.append(row)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="artifacts/datasets/static_v1")
    p.add_argument("--output-dir", default="artifacts/overfit_calibration")
    p.add_argument("--normalizations", default="window_zscore,dataset_zscore,none")
    p.add_argument("--seeds", default="123,223,323")
    p.add_argument("--linear-c", default="0.01,0.1,1.0,10.0")
    p.add_argument("--dimension", type=int, default=1000)
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--clip-z", type=float, default=3.0)
    p.add_argument("--target-val-fpr", type=float, default=0.20)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    summary = run(args)
    print(yaml.safe_dump(summary["artifacts"], sort_keys=False), end="")


if __name__ == "__main__":
    main()
