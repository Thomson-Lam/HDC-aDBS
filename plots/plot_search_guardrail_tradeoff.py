from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import (
    ARTIFACTS,
    read_csv_if_exists,
    read_yaml_if_exists,
    save_current,
    set_plot_style,
)


def generate(output_dir: Path) -> Path | None:
    df = read_csv_if_exists(ARTIFACTS / "encoder_search" / "leaderboard.csv")
    freeze = read_yaml_if_exists(ARTIFACTS / "encoder_search" / "freeze_record.yaml")
    if df is None or df.empty:
        print("[skip] leaderboard.csv not found")
        return None

    min_transition = 0.60
    max_holdout = 0.40
    if freeze is not None:
        cfg = freeze.get("search_config", {})
        min_transition = float(cfg.get("min_guardrail_auroc", min_transition))
        max_holdout = float(cfg.get("max_holdout_false_trigger_rate", max_holdout))

    set_plot_style()
    fig, ax = plt.subplots()
    sc = ax.scatter(
        df["healthy_holdout_false_trigger_rate"],
        df["min_transition_auroc"],
        c=df["balanced_accuracy_clean"],
        cmap="viridis",
        s=70,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.25,
    )
    ax.axvline(max_holdout, color="#d62828", linestyle="--", linewidth=1.2)
    ax.axhline(min_transition, color="#d62828", linestyle="--", linewidth=1.2)
    ax.set_title("Guardrail Tradeoff: Holdout FPR vs Min Transition AUROC")
    ax.set_xlabel("Healthy Holdout False Trigger Rate")
    ax.set_ylabel("Min( Onset AUROC, Recovery AUROC )")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Clean Balanced Accuracy")
    out = output_dir / "search_guardrail_tradeoff.png"
    return save_current(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/plots")
    args = parser.parse_args()
    out = generate(Path(args.output_dir))
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
