"""Offline validator search engine for HDC encoder candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from hdc.encoder import EncoderConfig, WindowEncoder
from hdc.readouts import LinearReadout, PrototypeReadout
from hdc.search.config import EncoderSearchSpec, SearchConfig


@dataclass(frozen=True)
class SplitData:
    """Window/label container for one split."""

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ValidationData:
    """All data needed for frozen validator ranking."""

    train: SplitData
    val_clean: SplitData
    val_onset: SplitData
    val_recovery: SplitData
    val_moderate: SplitData
    holdout_healthy: SplitData


@dataclass(frozen=True)
class SearchResult:
    """Metrics for one encoder + readout candidate."""

    dimension: int
    n_bins: int
    value_init: str
    readout: str
    balanced_accuracy_clean: float
    auroc_onset: float
    auroc_recovery: float
    auroc_moderate: float
    min_transition_auroc: float
    healthy_holdout_false_trigger_rate: float
    guardrail_pass: bool
    encoding_ms_per_window: float
    model_bytes: int
    rank: int | None = None


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    labels = np.unique(y_true)
    if labels.shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _estimate_model_bytes(readout_name: str, model: object) -> int:
    if readout_name == "prototype":
        proto = model
        return int(proto.healthy_prototype.nbytes + proto.path_prototype.nbytes)
    if readout_name == "linear":
        lin = model
        return int(lin.model.coef_.nbytes + lin.model.intercept_.nbytes)
    raise ValueError(f"unsupported readout: {readout_name}")


def _build_readout(name: str, seed: int) -> object:
    if name == "prototype":
        return PrototypeReadout()
    if name == "linear":
        return LinearReadout(seed=seed)
    raise ValueError(f"unsupported readout: {name}")


def _encode_with_timing(
    encoder: WindowEncoder, windows: np.ndarray
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    encoded = encoder.encode_batch(windows)
    elapsed = time.perf_counter() - start
    ms_per_window = (elapsed / max(1, windows.shape[0])) * 1000.0
    return encoded, float(ms_per_window)


def _score_candidate(
    spec: EncoderSearchSpec,
    cfg: SearchConfig,
    data: ValidationData,
    encoder_seed: int,
    readout_seed: int,
) -> list[SearchResult]:
    encoder = WindowEncoder(
        EncoderConfig(
            dimension=spec.dimension,
            n_bins=spec.n_bins,
            window_length=cfg.window_length,
            value_init=spec.value_init,
            clip_z=cfg.clip_z,
            seed=encoder_seed,
        )
    )

    x_train, train_ms = _encode_with_timing(encoder, data.train.x)
    x_clean, clean_ms = _encode_with_timing(encoder, data.val_clean.x)
    x_onset, onset_ms = _encode_with_timing(encoder, data.val_onset.x)
    x_recovery, recovery_ms = _encode_with_timing(encoder, data.val_recovery.x)
    x_moderate, moderate_ms = _encode_with_timing(encoder, data.val_moderate.x)
    x_holdout, holdout_ms = _encode_with_timing(encoder, data.holdout_healthy.x)
    mean_encode_ms = float(
        np.mean([train_ms, clean_ms, onset_ms, recovery_ms, moderate_ms, holdout_ms])
    )

    results: list[SearchResult] = []
    for readout_name in cfg.readouts:
        model = _build_readout(readout_name, seed=readout_seed)
        model.fit(x_train, data.train.y)

        clean_scores = model.decision_function(x_clean)
        clean_pred = model.predict(x_clean)
        onset_scores = model.decision_function(x_onset)
        recovery_scores = model.decision_function(x_recovery)
        moderate_scores = model.decision_function(x_moderate)
        holdout_pred = model.predict(x_holdout)

        balanced_acc = float(balanced_accuracy_score(data.val_clean.y, clean_pred))
        onset_auroc = _safe_auroc(data.val_onset.y, onset_scores)
        recovery_auroc = _safe_auroc(data.val_recovery.y, recovery_scores)
        moderate_auroc = _safe_auroc(data.val_moderate.y, moderate_scores)
        min_transition = float(np.nanmin(np.array([onset_auroc, recovery_auroc])))
        holdout_false_trigger_rate = float(np.mean(holdout_pred == 1))
        guardrail_pass = bool(
            min_transition >= cfg.min_guardrail_auroc
            and moderate_auroc >= cfg.min_moderate_auroc
            and holdout_false_trigger_rate <= cfg.max_holdout_false_trigger_rate
        )
        model_bytes = _estimate_model_bytes(readout_name, model)

        results.append(
            SearchResult(
                dimension=spec.dimension,
                n_bins=spec.n_bins,
                value_init=spec.value_init,
                readout=readout_name,
                balanced_accuracy_clean=balanced_acc,
                auroc_onset=onset_auroc,
                auroc_recovery=recovery_auroc,
                auroc_moderate=moderate_auroc,
                min_transition_auroc=min_transition,
                healthy_holdout_false_trigger_rate=holdout_false_trigger_rate,
                guardrail_pass=guardrail_pass,
                encoding_ms_per_window=mean_encode_ms,
                model_bytes=model_bytes,
            )
        )
    return results


def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    """Rank candidates by frozen rule with guardrail filtering."""
    eligible = [r for r in results if r.guardrail_pass]
    ineligible = [r for r in results if not r.guardrail_pass]

    eligible_sorted = sorted(
        eligible,
        key=lambda r: (-r.balanced_accuracy_clean, r.encoding_ms_per_window),
    )
    ineligible_sorted = sorted(
        ineligible,
        key=lambda r: (-r.balanced_accuracy_clean, r.encoding_ms_per_window),
    )

    ranked: list[SearchResult] = []
    for idx, row in enumerate(eligible_sorted, start=1):
        ranked.append(SearchResult(**{**asdict(row), "rank": idx}))
    for row in ineligible_sorted:
        ranked.append(SearchResult(**{**asdict(row), "rank": None}))
    return ranked


def run_validator_search(
    data: ValidationData, cfg: SearchConfig | None = None
) -> list[SearchResult]:
    """Run full fixed-grid validator search and return ranked rows."""
    cfg = cfg or SearchConfig()
    all_results: list[SearchResult] = []

    for idx, spec in enumerate(cfg.iter_encoder_specs()):
        encoder_seed = cfg.base_seed + (idx * 17)
        readout_seed = cfg.base_seed + (idx * 17) + 1
        all_results.extend(
            _score_candidate(
                spec=spec,
                cfg=cfg,
                data=data,
                encoder_seed=encoder_seed,
                readout_seed=readout_seed,
            )
        )
    return rank_results(all_results)


def top_candidates(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Return the top-k eligible rows after ranking."""
    ranked = [r for r in results if r.rank is not None]
    ranked = sorted(ranked, key=lambda r: r.rank)
    return ranked[:top_k]


def write_results_jsonl(results: list[SearchResult], output_path: str | Path) -> None:
    """Write search rows as JSON lines."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_results_csv(results: list[SearchResult], output_path: str | Path) -> None:
    """Write search rows as CSV leaderboard."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
