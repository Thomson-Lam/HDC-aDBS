"""Validator + trainer entrypoint with fixed search settings.

Usage:
    uv run python train/valid-train.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdc.encoder import EncoderConfig
from hdc.search.config import SearchConfig
from hdc.search.run import make_dummy_validation_data
from hdc.search.validator import (
    run_validator_search,
    top_candidates,
    write_results_csv,
    write_results_jsonl,
)
from hdc.training import LinearHDCTrainer, PrototypeHDCTrainer


ARTIFACT_ROOT = Path("artifacts")
SEARCH_DIR = ARTIFACT_ROOT / "encoder_search"
FREEZE_PATH = SEARCH_DIR / "freeze_record.yaml"
MODELS_DIR = ARTIFACT_ROOT / "models"


def ensure_freeze_record(cfg: SearchConfig) -> dict:
    """Load frozen selection or run validator search to create it."""
    if FREEZE_PATH.exists():
        return yaml.safe_load(FREEZE_PATH.read_text(encoding="utf-8"))

    data = make_dummy_validation_data(window_length=cfg.window_length)
    results = run_validator_search(data=data, cfg=cfg)
    write_results_jsonl(results, SEARCH_DIR / "results.jsonl")
    write_results_csv(results, SEARCH_DIR / "leaderboard.csv")

    winners = top_candidates(results, top_k=1)
    if not winners:
        raise RuntimeError("no eligible candidates passed guardrail")

    winner = winners[0]
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
            "min_transition_auroc": winner.min_transition_auroc,
            "encoding_ms_per_window": winner.encoding_ms_per_window,
            "model_bytes": winner.model_bytes,
            "rank": winner.rank,
        },
        "search_config": asdict(cfg),
    }
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    return freeze


def train_both_methods(freeze: dict, cfg: SearchConfig) -> None:
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

    data = make_dummy_validation_data(seed=3030, window_length=cfg.window_length)

    prototype = PrototypeHDCTrainer(encoder_config=encoder_config).fit(
        data.train.x, data.train.y
    )
    linear = LinearHDCTrainer(encoder_config=encoder_config, seed=cfg.base_seed).fit(
        data.train.x, data.train.y
    )

    proto_metrics = prototype.evaluate(data.val_clean.x, data.val_clean.y)
    linear_metrics = linear.evaluate(data.val_clean.x, data.val_clean.y)

    proto_path = MODELS_DIR / "prototype"
    linear_path = MODELS_DIR / "linear"
    prototype.save(proto_path)
    linear.save(linear_path)

    # Save a compact report for quick inspection.
    report = {
        "prototype": proto_metrics,
        "linear": linear_metrics,
        "encoder_config": asdict(encoder_config),
        "model_paths": {"prototype": str(proto_path), "linear": str(linear_path)},
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "train_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
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
    print(f"Prototype load-check max score delta: {float(proto_delta):.12f}")
    print(f"Linear load-check max score delta: {float(linear_delta):.12f}")
    print(f"Freeze record: {FREEZE_PATH}")
    print(f"Train report: {MODELS_DIR / 'train_report.yaml'}")


def main() -> None:
    cfg = SearchConfig()
    freeze = ensure_freeze_record(cfg)
    train_both_methods(freeze=freeze, cfg=cfg)


if __name__ == "__main__":
    main()
