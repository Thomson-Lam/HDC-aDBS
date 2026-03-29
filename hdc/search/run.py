"""Run fixed encoder validator search with built-in dummy data.

Usage:
    uv run python -m hdc.search.run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hdc.search.config import SearchConfig
from hdc.search.validator import (
    SplitData,
    ValidationData,
    run_validator_search,
    top_candidates,
    write_results_csv,
    write_results_jsonl,
)


def make_dummy_validation_data(
    seed: int = 2026, window_length: int = 128
) -> ValidationData:
    """Create synthetic data for validator smoke-testing and reproducibility."""
    rng = np.random.default_rng(seed)
    t = np.arange(window_length, dtype=np.float64) / 250.0
    beta = np.sin(2.0 * np.pi * 20.0 * t)

    def make_split(n_healthy: int, n_path: int, path_gain: float) -> SplitData:
        healthy = 0.85 * rng.standard_normal((n_healthy, window_length))
        pathological = (
            0.85 * rng.standard_normal((n_path, window_length)) + path_gain * beta
        )
        x = np.vstack([healthy, pathological])
        y = np.concatenate(
            [np.zeros(n_healthy, dtype=np.int64), np.ones(n_path, dtype=np.int64)]
        )
        return SplitData(x=x, y=y)

    return ValidationData(
        train=make_split(n_healthy=450, n_path=450, path_gain=1.20),
        val_clean=make_split(n_healthy=160, n_path=160, path_gain=1.20),
        val_onset=make_split(n_healthy=120, n_path=120, path_gain=0.70),
        val_recovery=make_split(n_healthy=120, n_path=120, path_gain=0.65),
    )


def main() -> None:
    cfg = SearchConfig()
    data = make_dummy_validation_data(window_length=cfg.window_length)

    results = run_validator_search(data=data, cfg=cfg)
    out_dir = Path("artifacts/encoder_search")
    write_results_jsonl(results, out_dir / "results.jsonl")
    write_results_csv(results, out_dir / "leaderboard.csv")

    winners = top_candidates(results, top_k=cfg.top_k)
    print("Top candidates:")
    for row in winners:
        print(
            f"rank={row.rank} D={row.dimension} bins={row.n_bins} init={row.value_init} "
            f"readout={row.readout} bal_acc={row.balanced_accuracy_clean:.4f} "
            f"min_transition_auroc={row.min_transition_auroc:.4f} "
            f"encode_ms={row.encoding_ms_per_window:.4f}"
        )

    print(f"Wrote: {out_dir / 'results.jsonl'}")
    print(f"Wrote: {out_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
