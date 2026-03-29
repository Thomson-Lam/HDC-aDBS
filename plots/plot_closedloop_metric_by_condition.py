from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ARTIFACTS, read_csv_if_exists, save_current, set_plot_style


def generate(output_dir: Path, metric: str = "pathological_occupancy") -> Path | None:
    df = read_csv_if_exists(ARTIFACTS / "closed_loop" / "summary_by_condition.csv")
    if df is None or df.empty:
        print(
            "[skip] summary_by_condition.csv not found (run closed-loop benchmark first)"
        )
        return None

    sub = df.copy()
    if "regime" in sub.columns:
        sub = sub[sub["regime"] == "pathological"]
    if sub.empty:
        print("[skip] no pathological rows in summary_by_condition.csv")
        return None

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in sub.columns:
        print(f"[skip] column {mean_col} missing")
        return None

    order = ["no_stimulation", "continuous_dbs", "beta_adbs", "hdc_adbs"]
    grouped = sub.set_index("condition").reindex(order)
    vals = grouped[mean_col].to_numpy(dtype=np.float64)
    errs = np.zeros_like(vals)
    if std_col in grouped.columns:
        errs = np.nan_to_num(grouped[std_col].to_numpy(dtype=np.float64), nan=0.0)

    labels = ["none", "continuous", "beta", "hdc"]
    set_plot_style()
    fig, ax = plt.subplots()
    x = np.arange(len(labels), dtype=np.float64)
    ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=4,
        color=["#9e9e9e", "#4c78a8", "#f58518", "#54a24b"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Closed-Loop {metric.replace('_', ' ').title()} by Condition")
    out = output_dir / f"closedloop_{metric}_by_condition.png"
    return save_current(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/plots")
    parser.add_argument("--metric", default="pathological_occupancy")
    args = parser.parse_args()
    out = generate(Path(args.output_dir), metric=args.metric)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
