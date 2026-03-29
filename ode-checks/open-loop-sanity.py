"""
Thin Runner code to run the frozen open-loop stimulation sanity gate, uses 
src/simulation/open_loop_sanity.py.

Usage:
    uv run python ode-checks/open-loop-sanity.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.open_loop_sanity import (
    OpenLoopGateConfig,
    run_open_loop_sanity_gate,
)


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("open_loop_sanity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def main() -> None:
    out_dir = Path("artifacts/open_loop_sanity")
    log_path = out_dir / "open_loop_sanity.log"
    logger = _configure_logger(log_path)

    cfg = OpenLoopGateConfig()
    logger.info("Starting open-loop sanity gate")
    logger.info("Config: %s", cfg)

    summary = run_open_loop_sanity_gate(config=cfg, output_dir=out_dir)

    logger.info("Gate pass: %s", summary["gate_pass"])
    for name, status in summary["checks"].items():
        logger.info("Check %s: %s", name, status)

    logger.info("Completed seeds: %s", summary["counts"]["n_seeds_completed"])
    logger.info("Artifacts: %s", summary["artifacts"])
    logger.info("Summary YAML: %s", out_dir / "summary.yaml")
    logger.info("Single log file: %s", log_path)


if __name__ == "__main__":
    main()
