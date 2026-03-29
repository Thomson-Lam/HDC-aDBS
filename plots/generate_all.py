from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plots.common import ensure_dir
from plots.plot_calibration_effect import generate as generate_calibration_effect
from plots.plot_closedloop_latency import generate as generate_closedloop_latency
from plots.plot_closedloop_metric_by_condition import (
    generate as generate_closedloop_metric,
)
from plots.plot_offline_model_performance import generate as generate_offline_perf
from plots.plot_overfit_gap import generate as generate_overfit_gap
from plots.plot_overfit_tradeoff import generate as generate_overfit_tradeoff
from plots.plot_search_guardrail_tradeoff import generate as generate_search_guardrails
from plots.plot_search_heatmap import generate as generate_search_heatmap
from plots.plot_search_speed_accuracy import generate as generate_search_speed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reporting plots from artifacts"
    )
    parser.add_argument("--output-dir", default="artifacts/plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    generated: list[Path] = []
    for fn in (
        generate_search_speed,
        generate_search_guardrails,
        generate_search_heatmap,
        generate_offline_perf,
        generate_calibration_effect,
        generate_overfit_tradeoff,
        generate_overfit_gap,
        generate_closedloop_latency,
    ):
        out = fn(out_dir)
        if out is not None:
            generated.append(out)

    for metric in ("pathological_occupancy", "duty_cycle", "mean_beta_power"):
        out = generate_closedloop_metric(out_dir, metric=metric)
        if out is not None:
            generated.append(out)

    if generated:
        print("Generated plots:")
        for path in generated:
            print(path)
    else:
        print("No plots generated (required input artifacts missing).")


if __name__ == "__main__":
    main()
