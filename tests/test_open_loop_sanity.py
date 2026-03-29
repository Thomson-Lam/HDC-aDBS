from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.simulation.open_loop_sanity import (
    OpenLoopGateConfig,
    make_pulse_train_stim,
    run_open_loop_sanity_gate,
)


class TestOpenLoopSanityHelpers(unittest.TestCase):
    def test_pulse_train_function_outputs_expected_levels(self) -> None:
        stim = make_pulse_train_stim(
            amplitude=5.0, frequency_hz=100.0, pulse_width_ms=1.0
        )
        self.assertEqual(stim(0.0), 5.0)
        self.assertEqual(stim(0.5), 5.0)
        self.assertEqual(stim(1.5), 0.0)
        self.assertEqual(stim(10.0), 5.0)


class TestOpenLoopSanityGate(unittest.TestCase):
    def test_gate_runs_and_writes_artifacts(self) -> None:
        cfg = OpenLoopGateConfig(
            seeds=(0,),
            t_end_ms=1200.0,
            t_warmup_ms=500.0,
            min_seed_pass=1,
            weak_amplitude=1.0,
            strong_amplitude=8.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "open_loop"
            summary = run_open_loop_sanity_gate(config=cfg, output_dir=out_dir)

            self.assertIn("gate_pass", summary)
            self.assertIn("checks", summary)
            self.assertIn("artifacts", summary)
            self.assertEqual(summary["counts"]["n_seeds_completed"], 1)

            self.assertTrue((out_dir / "summary.yaml").exists())
            self.assertTrue((out_dir / "per_seed_metrics.csv").exists())
            self.assertTrue((out_dir / "seed0_traces.png").exists())
            self.assertTrue((out_dir / "beta_summary.png").exists())


if __name__ == "__main__":
    unittest.main()
