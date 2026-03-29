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


def generate(output_dir: Path) -> Path | None:
    df = read_csv_if_exists(ARTIFACTS / "closed_loop" / "per_run_metrics.csv")
    if df is None or df.empty:
        print("[skip] per_run_metrics.csv not found (run closed-loop benchmark first)")
        return None

    sub = df[df["condition"].isin(["beta_adbs", "hdc_adbs"])].copy()
    if "regime" in sub.columns:
        sub = sub[sub["regime"] == "pathological"]
    if sub.empty:
        print("[skip] no beta/hdc pathological rows in closed-loop metrics")
        return None

    mean_col = "decision_time_mean_ms"
    if mean_col not in sub.columns:
        print("[skip] decision_time_mean_ms missing in closed-loop metrics")
        return None

    grouped = (
        sub.groupby("condition")[mean_col]
        .agg(["mean", "std"])
        .reindex(["beta_adbs", "hdc_adbs"])
    )
    labels = ["beta threshold", "hdc"]
    vals = grouped["mean"].to_numpy(dtype=np.float64)
    errs = np.nan_to_num(grouped["std"].to_numpy(dtype=np.float64), nan=0.0)

    set_plot_style()
    fig, ax = plt.subplots()
    x = np.arange(len(labels), dtype=np.float64)
    ax.bar(x, vals, yerr=errs, capsize=4, color=["#4c78a8", "#f58518"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Decision Time Mean (ms)")
    ax.set_title("Closed-Loop Inference Latency: Beta vs HDC")
    out = output_dir / "closedloop_decision_latency.png"
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
