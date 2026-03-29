"""Tests for pulse waveform and closed-loop benchmark metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controllers.eval_metrics import decision_time_stats, pulse_count_from_stim
from controllers.run_closedloop_benchmark import run_benchmark
from controllers.waveform import make_epoch_gated_pulse_train, pulse_train_value


def test_pulse_train_value_periodic():
    amp = 2.0
    f = 100.0  # period = 10 ms
    pw = 1.0
    assert pulse_train_value(0.0, amp, f, pw) == amp
    assert pulse_train_value(0.5, amp, f, pw) == amp
    assert pulse_train_value(1.5, amp, f, pw) == 0.0
    assert pulse_train_value(10.0, amp, f, pw) == amp


def test_epoch_gated_pulse_train():
    stim_fn = make_epoch_gated_pulse_train(
        amplitude=1.5,
        frequency_hz=100.0,
        pulse_width_ms=1.0,
        epoch_active_fn=lambda t: t < 20.0,
    )
    assert stim_fn(0.0) == 1.5
    assert stim_fn(0.5) == 1.5
    assert stim_fn(10.5) == 1.5
    assert stim_fn(6.5) == 0.0
    assert stim_fn(21.0) == 0.0


def test_pulse_count_from_stim_counts_onsets():
    stim = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0])
    assert pulse_count_from_stim(stim) == 3


def test_decision_time_stats_returns_mean_and_p95():
    stats = decision_time_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean_ms"] == 3.0
    assert stats["p95_ms"] >= 4.0
    assert stats["max_ms"] == 5.0


def test_closedloop_benchmark_smoke(tmp_path: Path):
    out_dir = tmp_path / "closed_loop"
    args = argparse.Namespace(
        model_path="artifacts/models/linear",
        trainer_type="linear",
        output_dir=str(out_dir),
        seeds="0",
        healthy_seeds="100",
        t_end_ms=1200.0,
        t_warmup_ms=500.0,
        stim_amplitude=1.5,
        beta_threshold=0.0,
        hdc_threshold=0.0,
        pathology_percentile=50.0,
    )
    summary = run_benchmark(args)

    assert (out_dir / "per_run_metrics.csv").exists()
    assert (out_dir / "summary_by_condition.csv").exists()
    assert (out_dir / "summary.yaml").exists()
    assert (out_dir / "run_manifest.yaml").exists()

    assert "artifacts" in summary
    assert set(summary["conditions"]) == {
        "no_stimulation",
        "continuous_dbs",
        "beta_adbs",
        "hdc_adbs",
    }
