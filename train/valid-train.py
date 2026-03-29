"""

Validator + trainer entrypoint with fixed search settings. Uses hdc/ definitions.

Usage:
    uv run python train/valid-train.py
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdc.encoder import EncoderConfig
from hdc.search.config import SearchConfig
from hdc.search.validator import (
    ValidationData,
    run_validator_search,
    top_candidates,
    write_results_csv,
    write_results_jsonl,
)
from hdc.training import LinearHDCTrainer, PrototypeHDCTrainer
from src.data.hdc_adapter import (
    ensure_static_dataset_ready,
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
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    healthy = y == 0
    pathological = y == 1
    fpr = float(np.mean(pred[healthy] == 1)) if np.any(healthy) else float("nan")
    tpr = (
        float(np.mean(pred[pathological] == 1))
        if np.any(pathological)
        else float("nan")
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fpr": fpr,
        "tpr": tpr,
        "auroc": _safe_auroc(y, scores),
    }


def _calibrate_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    target_val_fpr: float,
) -> tuple[float, dict[str, float]]:
    unique_thresholds = np.unique(scores)
    candidates = np.concatenate(
        (
            np.array([float("-inf")], dtype=np.float64),
            unique_thresholds,
            np.array([float("inf")], dtype=np.float64),
        )
    )

    best_threshold = 0.0
    best_stats = {
        "balanced_accuracy": float("-inf"),
        "fpr": float("nan"),
        "tpr": float("nan"),
        "auroc": float("nan"),
    }
    feasible = False

    for threshold in candidates:
        stats = _threshold_stats(scores, y, float(threshold))
        if not np.isnan(stats["fpr"]) and stats["fpr"] <= target_val_fpr:
            feasible = True
            if stats["balanced_accuracy"] > best_stats["balanced_accuracy"]:
                best_threshold = float(threshold)
                best_stats = stats

    if feasible:
        return best_threshold, best_stats

    # Fallback: minimize FPR first, then maximize balanced accuracy.
    ranked: list[tuple[float, float, float, dict[str, float]]] = []
    for threshold in candidates:
        stats = _threshold_stats(scores, y, float(threshold))
        ranked.append(
            (stats["fpr"], -stats["balanced_accuracy"], float(threshold), stats)
        )
    ranked.sort(key=lambda x: (x[0], x[1]))
    _, _, best_threshold, best_stats = ranked[0]
    return float(best_threshold), best_stats


def _calibrated_metrics_for_splits(
    *,
    trainer,
    threshold: float,
    data: ValidationData,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    train_scores = trainer.decision_function(data.train.x)
    clean_scores = trainer.decision_function(data.val_clean.x)
    onset_scores = trainer.decision_function(data.val_onset.x)
    recovery_scores = trainer.decision_function(data.val_recovery.x)
    moderate_scores = trainer.decision_function(data.val_moderate.x)
    holdout_scores = trainer.decision_function(data.holdout_healthy.x)

    out: dict[str, object] = {
        "train": _threshold_stats(train_scores, data.train.y, threshold),
        "val_clean": _threshold_stats(clean_scores, data.val_clean.y, threshold),
        "val_onset": _threshold_stats(onset_scores, data.val_onset.y, threshold),
        "val_recovery": _threshold_stats(
            recovery_scores, data.val_recovery.y, threshold
        ),
        "val_moderate": _threshold_stats(
            moderate_scores, data.val_moderate.y, threshold
        ),
        "holdout_false_trigger_rate": float(
            np.mean((holdout_scores >= threshold).astype(np.int64) == 1)
        ),
    }

    if x_test.shape[0] > 0 and y_test.shape[0] > 0:
        test_scores = trainer.decision_function(x_test)
        out["test_clean"] = _threshold_stats(test_scores, y_test, threshold)
    else:
        out["test_clean"] = {
            "balanced_accuracy": float("nan"),
            "fpr": float("nan"),
            "tpr": float("nan"),
            "auroc": float("nan"),
        }

    return out


ARTIFACT_ROOT = Path("artifacts")
SEARCH_DIR = ARTIFACT_ROOT / "encoder_search"
FREEZE_PATH = SEARCH_DIR / "freeze_record.yaml"
MODELS_DIR = ARTIFACT_ROOT / "models"
DATASET_DIR = ARTIFACT_ROOT / "datasets" / "static_v1"


def _label_counts(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y.astype(np.int64), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(labels, counts)}


def _safe_split_scenario_counts(dataset_dir: Path) -> list[dict[str, object]]:
    split_manifest = dataset_dir / "manifest_with_splits.csv"
    if not split_manifest.exists():
        return []
    df = pd.read_csv(split_manifest)
    if "split" not in df.columns or "scenario" not in df.columns:
        return []
    grouped = (
        df.groupby(["split", "scenario"]).size().reset_index(name="n_trajectories")
    )
    return grouped.to_dict(orient="records")


def write_training_audit(
    *,
    cfg: SearchConfig,
    freeze: dict,
    data: ValidationData,
    report: dict,
    output_path: Path,
    dataset_dir: Path,
) -> None:
    """Write one auditable training record with data coverage and guardrails."""
    selection_metrics = freeze.get("selection_metrics", {})
    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(dataset_dir),
        "artifacts": {
            "freeze_record": str(FREEZE_PATH),
            "search_leaderboard": str(SEARCH_DIR / "leaderboard.csv"),
            "search_results": str(SEARCH_DIR / "results.jsonl"),
            "train_report": str(MODELS_DIR / "train_report.yaml"),
        },
        "guardrail_thresholds": {
            "min_transition_auroc": float(cfg.min_guardrail_auroc),
            "min_moderate_auroc": float(cfg.min_moderate_auroc),
            "max_holdout_false_trigger_rate": float(cfg.max_holdout_false_trigger_rate),
        },
        "selected_config": freeze.get("selected", {}),
        "selected_metrics": selection_metrics,
        "split_scenario_trajectory_counts": _safe_split_scenario_counts(dataset_dir),
        "window_coverage": {
            "train": {
                "n_windows": int(data.train.x.shape[0]),
                "label_counts": _label_counts(data.train.y),
            },
            "val_clean": {
                "n_windows": int(data.val_clean.x.shape[0]),
                "label_counts": _label_counts(data.val_clean.y),
            },
            "val_onset": {
                "n_windows": int(data.val_onset.x.shape[0]),
                "label_counts": _label_counts(data.val_onset.y),
            },
            "val_recovery": {
                "n_windows": int(data.val_recovery.x.shape[0]),
                "label_counts": _label_counts(data.val_recovery.y),
            },
            "val_moderate": {
                "n_windows": int(data.val_moderate.x.shape[0]),
                "label_counts": _label_counts(data.val_moderate.y),
            },
            "holdout_healthy": {
                "n_windows": int(data.holdout_healthy.x.shape[0]),
                "label_counts": _label_counts(data.holdout_healthy.y),
            },
        },
        "reported_metrics": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(audit, sort_keys=False), encoding="utf-8")


def ensure_freeze_record(cfg: SearchConfig, data: ValidationData) -> dict:
    """Load frozen selection or run validator search to create it."""
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))

    results = run_validator_search(data=data, cfg=cfg)
    write_results_jsonl(results, SEARCH_DIR / "results.jsonl")
    write_results_csv(results, SEARCH_DIR / "leaderboard.csv")

    winners = top_candidates(results, top_k=1)
    fallback_used = False
    if winners:
        winner = winners[0]
    else:
        if not cfg.allow_guardrail_fallback:
            raise RuntimeError("no eligible candidates passed guardrail")
        if not results:
            raise RuntimeError("validator search returned no candidates")
        winner = sorted(
            results,
            key=lambda r: (-r.balanced_accuracy_clean, r.encoding_ms_per_window),
        )[0]
        fallback_used = True
        print(
            "WARNING: no candidates passed guardrails; "
            f"falling back to best available candidate "
            f"(D={winner.dimension}, bins={winner.n_bins}, init={winner.value_init}, "
            f"readout={winner.readout})."
        )
    freeze = {
        "selected": {
            "dimension": winner.dimension,
            "n_bins": winner.n_bins,
            "value_init": winner.value_init,
            "readout": winner.readout,
            "clip_z": cfg.clip_z,
            "seed": cfg.base_seed,
        },
        "selection_metrics": {
            "balanced_accuracy_clean": winner.balanced_accuracy_clean,
            "auroc_onset": winner.auroc_onset,
            "auroc_recovery": winner.auroc_recovery,
            "auroc_moderate": winner.auroc_moderate,
            "min_transition_auroc": winner.min_transition_auroc,
            "healthy_holdout_false_trigger_rate": winner.healthy_holdout_false_trigger_rate,
            "guardrail_pass": bool(winner.guardrail_pass),
            "encoding_ms_per_window": winner.encoding_ms_per_window,
            "model_bytes": winner.model_bytes,
            "rank": winner.rank,
        },
        "search_config": asdict(cfg),
    }
    if fallback_used:
        freeze["selection_fallback"] = {
            "used": True,
            "reason": "no eligible candidates passed guardrail",
            "guardrail_thresholds": {
                "min_transition_auroc": float(cfg.min_guardrail_auroc),
                "min_moderate_auroc": float(cfg.min_moderate_auroc),
                "max_holdout_false_trigger_rate": float(
                    cfg.max_holdout_false_trigger_rate
                ),
            },
        }
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    return freeze


def train_both_methods(freeze: dict, cfg: SearchConfig, data: ValidationData) -> None:
    """Train prototype and linear trainers from one frozen encoder config."""
    selection = freeze["selected"]
    encoder_config = EncoderConfig(
        dimension=int(selection["dimension"]),
        n_bins=int(selection["n_bins"]),
        window_length=cfg.window_length,
        value_init=str(selection["value_init"]),
        clip_z=float(selection["clip_z"]),
        seed=int(selection["seed"]),
    )

    prototype = PrototypeHDCTrainer(encoder_config=encoder_config).fit(
        data.train.x, data.train.y
    )

    # Tune linear regularization C on clean validation using calibrated threshold.
    best_linear: LinearHDCTrainer | None = None
    best_linear_c = float("nan")
    best_linear_threshold = 0.0
    best_linear_stats = {
        "balanced_accuracy": float("-inf"),
        "fpr": float("nan"),
        "tpr": float("nan"),
        "auroc": float("nan"),
    }
    for c in cfg.linear_c_candidates:
        candidate = LinearHDCTrainer(
            encoder_config=encoder_config,
            seed=cfg.base_seed,
            c=float(c),
        ).fit(data.train.x, data.train.y)
        clean_scores = candidate.decision_function(data.val_clean.x)
        threshold, clean_stats = _calibrate_threshold(
            clean_scores,
            data.val_clean.y,
            target_val_fpr=float(cfg.calibration_target_val_fpr),
        )
        if clean_stats["balanced_accuracy"] > best_linear_stats["balanced_accuracy"]:
            best_linear = candidate
            best_linear_c = float(c)
            best_linear_threshold = float(threshold)
            best_linear_stats = clean_stats
    if best_linear is None:
        raise RuntimeError("failed to fit any linear candidate")
    linear = best_linear

    proto_metrics = prototype.evaluate(data.val_clean.x, data.val_clean.y)
    linear_metrics = linear.evaluate(data.val_clean.x, data.val_clean.y)
    proto_onset_metrics = prototype.evaluate(data.val_onset.x, data.val_onset.y)
    linear_onset_metrics = linear.evaluate(data.val_onset.x, data.val_onset.y)
    proto_recovery_metrics = prototype.evaluate(
        data.val_recovery.x, data.val_recovery.y
    )
    linear_recovery_metrics = linear.evaluate(data.val_recovery.x, data.val_recovery.y)
    proto_moderate_metrics = prototype.evaluate(
        data.val_moderate.x, data.val_moderate.y
    )
    linear_moderate_metrics = linear.evaluate(data.val_moderate.x, data.val_moderate.y)
    proto_holdout_pred = prototype.predict(data.holdout_healthy.x)
    linear_holdout_pred = linear.predict(data.holdout_healthy.x)
    proto_holdout_false_trigger_rate = float(np.mean(proto_holdout_pred == 1))
    linear_holdout_false_trigger_rate = float(np.mean(linear_holdout_pred == 1))

    # Optional held-out view from static split test windows (not used for selection).
    x_train_split, y_train_split, x_test_split, y_test_split = (
        load_train_and_test_windows(
            DATASET_DIR,
            window_cfg=WindowingConfig(
                window_length=cfg.window_length,
                stride_ms=float(cfg.stride_ms),
                sampling_rate_hz=float(cfg.sampling_rate_hz),
            ),
        )
    )
    if x_train_split.shape[0] > 0 and y_train_split.shape[0] > 0:
        # Keep both trainers fit on ValidationData train split; test-only reporting here.
        if x_test_split.shape[0] > 0 and y_test_split.shape[0] > 0:
            proto_test_metrics = prototype.evaluate(x_test_split, y_test_split)
            linear_test_metrics = linear.evaluate(x_test_split, y_test_split)
        else:
            proto_test_metrics = {
                "balanced_accuracy": float("nan"),
                "auroc": float("nan"),
            }
            linear_test_metrics = {
                "balanced_accuracy": float("nan"),
                "auroc": float("nan"),
            }
    else:
        proto_test_metrics = {"balanced_accuracy": float("nan"), "auroc": float("nan")}
        linear_test_metrics = {"balanced_accuracy": float("nan"), "auroc": float("nan")}

    # Calibrate score thresholds on validation clean split.
    proto_clean_scores = prototype.decision_function(data.val_clean.x)
    proto_threshold, proto_calibration = _calibrate_threshold(
        proto_clean_scores,
        data.val_clean.y,
        target_val_fpr=float(cfg.calibration_target_val_fpr),
    )
    linear_calibration = {
        "balanced_accuracy": best_linear_stats["balanced_accuracy"],
        "fpr": best_linear_stats["fpr"],
        "tpr": best_linear_stats["tpr"],
        "auroc": best_linear_stats["auroc"],
    }

    proto_calibrated = _calibrated_metrics_for_splits(
        trainer=prototype,
        threshold=proto_threshold,
        data=data,
        x_test=x_test_split,
        y_test=y_test_split,
    )
    linear_calibrated = _calibrated_metrics_for_splits(
        trainer=linear,
        threshold=best_linear_threshold,
        data=data,
        x_test=x_test_split,
        y_test=y_test_split,
    )

    proto_path = MODELS_DIR / "prototype"
    linear_path = MODELS_DIR / "linear"
    prototype.save(proto_path)
    linear.save(linear_path)

    # Save a compact report for quick inspection.
    report = {
        "prototype": proto_metrics,
        "linear": linear_metrics,
        "prototype_onset": proto_onset_metrics,
        "linear_onset": linear_onset_metrics,
        "prototype_recovery": proto_recovery_metrics,
        "linear_recovery": linear_recovery_metrics,
        "prototype_moderate": proto_moderate_metrics,
        "linear_moderate": linear_moderate_metrics,
        "prototype_holdout_false_trigger_rate": proto_holdout_false_trigger_rate,
        "linear_holdout_false_trigger_rate": linear_holdout_false_trigger_rate,
        "prototype_test": proto_test_metrics,
        "linear_test": linear_test_metrics,
        "calibration": {
            "target_val_fpr": float(cfg.calibration_target_val_fpr),
            "prototype": {
                "threshold": float(proto_threshold),
                "val_clean": proto_calibration,
            },
            "linear": {
                "selected_c": best_linear_c,
                "threshold": float(best_linear_threshold),
                "val_clean": linear_calibration,
            },
        },
        "prototype_calibrated": proto_calibrated,
        "linear_calibrated": linear_calibrated,
        "encoder_config": asdict(encoder_config),
        "model_paths": {"prototype": str(proto_path), "linear": str(linear_path)},
        "data_source": str(DATASET_DIR),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "train_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )

    audit_path = MODELS_DIR / "training_audit.yaml"
    write_training_audit(
        cfg=cfg,
        freeze=freeze,
        data=data,
        report=report,
        output_path=audit_path,
        dataset_dir=DATASET_DIR,
    )

    # Quick load-check for reproducible inference path.
    prototype_loaded = PrototypeHDCTrainer.load(proto_path)
    linear_loaded = LinearHDCTrainer.load(linear_path)
    proto_delta = np.max(
        np.abs(
            prototype.decision_function(data.val_clean.x)
            - prototype_loaded.decision_function(data.val_clean.x)
        )
    )
    linear_delta = np.max(
        np.abs(
            linear.decision_function(data.val_clean.x)
            - linear_loaded.decision_function(data.val_clean.x)
        )
    )

    print("Training complete")
    print(f"Prototype clean metrics: {proto_metrics}")
    print(f"Linear clean metrics: {linear_metrics}")
    print(
        "Prototype calibrated "
        f"(thr={proto_threshold:.6f}) val_bal={proto_calibration['balanced_accuracy']:.4f} "
        f"val_fpr={proto_calibration['fpr']:.4f} "
        f"holdout_fpr={proto_calibrated['holdout_false_trigger_rate']:.4f}"
    )
    print(
        "Linear calibrated "
        f"(C={best_linear_c:.4g}, thr={best_linear_threshold:.6f}) "
        f"val_bal={linear_calibration['balanced_accuracy']:.4f} "
        f"val_fpr={linear_calibration['fpr']:.4f} "
        f"holdout_fpr={linear_calibrated['holdout_false_trigger_rate']:.4f}"
    )
    print(f"Prototype load-check max score delta: {float(proto_delta):.12f}")
    print(f"Linear load-check max score delta: {float(linear_delta):.12f}")
    print(f"Freeze record: {FREEZE_PATH}")
    print(f"Train report: {MODELS_DIR / 'train_report.yaml'}")
    print(f"Training audit: {MODELS_DIR / 'training_audit.yaml'}")


def main() -> None:
    cfg = SearchConfig()
    ensure_static_dataset_ready(DATASET_DIR)
    data = load_validation_data_from_static(
        DATASET_DIR,
        window_cfg=WindowingConfig(
            window_length=cfg.window_length,
            stride_ms=float(cfg.stride_ms),
            sampling_rate_hz=float(cfg.sampling_rate_hz),
        ),
    )
    freeze = ensure_freeze_record(cfg=cfg, data=data)
    train_both_methods(freeze=freeze, cfg=cfg, data=data)


if __name__ == "__main__":
    main()
