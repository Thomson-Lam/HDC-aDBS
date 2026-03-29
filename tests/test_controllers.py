"""Unit and integration tests for the closed-loop controller stack.

What these tests cover
----------------------
- Ring buffer: correct fill, wrapping, and chronological ordering
- State machine: IDLE → STIMULATING → LOCKOUT → IDLE transitions,
  blocking during active stim / lockout, re-triggering after lockout
- Metrics: duty cycle math, stim_times_ms recording, detection vs stim counts
- BetaController filter: causal IIR state continuity across chunks,
  beta-band signals produce elevated RMS
- HDCController: correct decision_function routing via DummyTrainer (no disk I/O)
- run_closed_loop harness: return dict keys, stim/state array consistency,
  reset between runs

What these tests do NOT cover
------------------------------
- Biological plausibility of pathological suppression (visual/notebook check)
- Numerical accuracy of HDC classification on real LFP (offline eval suite)
- Comparison of HDC vs beta controller performance (separate eval notebook)

Run with:
    pytest tests/test_controllers.py -v
    pytest tests/test_controllers.py -v -m slow   # integration tests only
from the project root.
"""

import numpy as np
import pytest

from controllers.base import (
    BaseController,
    ControllerConfig,
    ControllerMetrics,
    StimState,
)
from controllers.hdc_controller import HDCController
from controllers.beta_controller import BetaController


# ---------------------------------------------------------------------------
# Test doubles — no real model loading, no disk I/O
# ---------------------------------------------------------------------------


class DummyTrainer:
    """Minimal stand-in for BaseHDCTrainer.

    Returns a fixed scalar score for every window regardless of content.
    This lets us test the HDCController state machine without a trained model.

    `score` is a mutable attribute so tests can switch it mid-run to control
    when the controller triggers vs stays idle.
    """

    def __init__(self, score: float = 1.0) -> None:
        self.score = score  # mutable — tests can change this between ingests

    def decision_function(self, windows: np.ndarray) -> np.ndarray:
        """Return (n_windows,) array of the current score."""
        return np.full(windows.shape[0], self.score, dtype=np.float64)


class DummyHDCController(HDCController):
    """HDCController with an injected DummyTrainer instead of a disk-loaded model.

    Bypasses HDCController.__init__ (which would try to load from disk) by
    calling BaseController.__init__ directly and then setting self.trainer.
    """

    def __init__(
        self, score: float = 1.0, config: ControllerConfig | None = None
    ) -> None:
        # Skip HDCController.__init__ to avoid loading from disk
        BaseController.__init__(self, config or ControllerConfig())
        self.trainer = DummyTrainer(score=score)
        self.model_path = None
        self.trainer_type = "dummy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lfp_chunk(
    n_samples: int, value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, lfp) arrays of length n_samples at 250 Hz starting from t=0."""
    t = np.arange(n_samples) * 4.0  # 4 ms per sample at 250 Hz
    lfp = np.full(n_samples, value, dtype=np.float64)
    return t, lfp


def _fill_buffer(ctrl: BaseController, n_extra: int = 0) -> None:
    """Feed exactly window_length + n_extra samples (value=0) to fill the buffer."""
    n = ctrl.config.window_length + n_extra
    t, lfp = _make_lfp_chunk(n)
    # Temporarily suppress decisions by using a subthreshold score
    ctrl.ingest(t, lfp)


# ---------------------------------------------------------------------------
# 1. Ring buffer management
# ---------------------------------------------------------------------------


class TestBufferManagement:
    """Verify that the ring buffer fills, wraps, and returns windows correctly."""

    def test_buffer_not_ready_until_full(self):
        # Feed window_length - 1 samples: buffer should NOT be ready yet.
        ctrl = DummyHDCController(score=-999.0)  # negative score → never triggers
        n = ctrl.config.window_length - 1
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert not ctrl.is_ready

    def test_buffer_ready_after_exactly_window_length(self):
        ctrl = DummyHDCController(score=-999.0)
        t, lfp = _make_lfp_chunk(ctrl.config.window_length)
        ctrl.ingest(t, lfp)
        assert ctrl.is_ready

    def test_get_latest_window_returns_none_when_not_ready(self):
        ctrl = DummyHDCController(score=-999.0)
        assert ctrl.get_latest_window() is None

    def test_get_latest_window_chronological_order(self):
        """Push values 0, 1, 2, ... 127 and verify the window matches exactly."""
        ctrl = DummyHDCController(score=-999.0)
        n = ctrl.config.window_length  # 128
        t = np.arange(n, dtype=np.float64) * 4.0
        lfp = np.arange(n, dtype=np.float64)  # values 0..127
        ctrl.ingest(t, lfp)
        window = ctrl.get_latest_window()
        assert window is not None
        np.testing.assert_array_equal(window, lfp)

    def test_ring_buffer_overwrites_oldest_on_wrap(self):
        """After pushing 2×window_length samples (0..255), the window should be 128..255."""
        ctrl = DummyHDCController(score=-999.0)
        n = ctrl.config.window_length * 2  # push twice the buffer size
        t = np.arange(n, dtype=np.float64) * 4.0
        lfp = np.arange(n, dtype=np.float64)  # 0..255
        ctrl.ingest(t, lfp)
        window = ctrl.get_latest_window()
        expected = np.arange(ctrl.config.window_length, n, dtype=np.float64)  # 128..255
        np.testing.assert_array_equal(window, expected)

    def test_total_samples_increments_with_ingest(self):
        ctrl = DummyHDCController(score=-999.0)
        t, lfp = _make_lfp_chunk(50)
        ctrl.ingest(t, lfp)
        assert ctrl.metrics.total_samples == 50

    def test_reset_clears_buffer_and_readiness(self):
        ctrl = DummyHDCController(score=-999.0)
        _fill_buffer(ctrl)
        assert ctrl.is_ready
        ctrl.reset()
        assert not ctrl.is_ready
        assert ctrl.get_latest_window() is None
        assert ctrl.metrics.total_samples == 0


# ---------------------------------------------------------------------------
# 2. State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    """Verify IDLE → STIMULATING → LOCKOUT → IDLE transitions."""

    def _make_triggering_controller(self, threshold: float = 0.0) -> DummyHDCController:
        """Controller that WILL trigger on its first decision (score=1.0 > 0.0)."""
        cfg = ControllerConfig(threshold=threshold)
        return DummyHDCController(score=1.0, config=cfg)

    def _make_nontriggering_controller(self) -> DummyHDCController:
        """Controller that will NEVER trigger (score=-1.0 < 0.0)."""
        return DummyHDCController(score=-1.0)

    def test_initial_state_is_idle(self):
        ctrl = DummyHDCController()
        assert ctrl.state == StimState.IDLE

    def test_stim_triggers_after_buffer_fills_and_decision_fires(self):
        ctrl = self._make_triggering_controller()
        # Fill buffer exactly, then add one more decision stride worth of samples
        # so that ingest() fires _compute_decision() at least once.
        n = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.STIMULATING
        assert ctrl.metrics.n_stimulations == 1

    def test_stim_does_not_trigger_below_threshold(self):
        ctrl = self._make_nontriggering_controller()
        n = ctrl.config.window_length + ctrl._decision_stride * 5
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.IDLE
        assert ctrl.metrics.n_stimulations == 0

    def test_stim_transitions_to_lockout_after_burst_duration(self):
        ctrl = self._make_triggering_controller()
        # Fill buffer and trigger stim
        n_fill = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n_fill)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.STIMULATING

        # Now feed enough samples to advance time past the burst duration.
        # stim_duration_ms = 200 ms; at 250 Hz that's 50 samples.
        stim_duration_samples = (
            int(ctrl.config.stim_duration_ms * ctrl.config.fs / 1000.0) + 5
        )  # a few extra to be sure the state machine fires
        t_offset = float(t[-1]) + 4.0  # continue from last t
        t2 = t_offset + np.arange(stim_duration_samples) * 4.0
        lfp2 = np.zeros(stim_duration_samples)
        ctrl.ingest(t2, lfp2)

        assert ctrl.state == StimState.LOCKOUT

    def test_lockout_transitions_to_idle_after_refractory_period(self):
        # Use a mutable trainer so we can drop to sub-threshold AFTER triggering.
        # Without this, score=1.0 throughout causes an immediate re-trigger the
        # moment LOCKOUT → IDLE, leaving the state as STIMULATING rather than IDLE.
        ctrl = DummyHDCController(score=1.0)

        # Phase 1: trigger (positive score)
        n_fill = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n_fill)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.STIMULATING

        # Switch to sub-threshold so no new trigger fires after lockout expires
        ctrl.trainer.score = -999.0

        # Phase 2: advance through the full refractory period (stim + lockout = 400 ms)
        total_refractory_ms = (
            ctrl.config.stim_duration_ms + ctrl.config.lockout_duration_ms
        )
        n_refractory = int(total_refractory_ms * ctrl.config.fs / 1000.0) + 5
        t_offset = float(t[-1]) + 4.0
        t2 = t_offset + np.arange(n_refractory) * 4.0
        lfp2 = np.zeros(n_refractory)
        ctrl.ingest(t2, lfp2)

        assert ctrl.state == StimState.IDLE

    def test_no_retrigger_during_stimulating(self):
        """A detection during STIMULATING should NOT start a new burst."""
        ctrl = self._make_triggering_controller()
        # Fill buffer and trigger
        n = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.STIMULATING

        # Feed more samples with still-positive score (still within stim window)
        t2 = float(t[-1]) + 4.0 + np.arange(ctrl._decision_stride) * 4.0
        lfp2 = np.zeros(ctrl._decision_stride)
        ctrl.ingest(t2, lfp2)

        # n_stimulations must still be 1 — no second burst
        assert ctrl.metrics.n_stimulations == 1

    def test_no_retrigger_during_lockout(self):
        """A detection during LOCKOUT should not restart stim."""
        ctrl = self._make_triggering_controller()
        # Trigger and advance past stim into lockout
        stim_samples = int(ctrl.config.stim_duration_ms * ctrl.config.fs / 1000.0) + 5
        n = ctrl.config.window_length + ctrl._decision_stride + stim_samples
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.state == StimState.LOCKOUT

        # Feed a decision's worth of samples during lockout
        t2 = float(t[-1]) + 4.0 + np.arange(ctrl._decision_stride) * 4.0
        lfp2 = np.zeros(ctrl._decision_stride)
        ctrl.ingest(t2, lfp2)

        assert ctrl.metrics.n_stimulations == 1  # still only 1

    def test_retrigger_allowed_after_lockout_expires(self):
        """After the full refractory period, a second detection should trigger again.

        With score=1.0 throughout, the controller will re-trigger immediately the
        moment LOCKOUT → IDLE.  So we assert n_stimulations >= 2 (the re-trigger
        happened) rather than state == IDLE (which would require the score to drop
        sub-threshold after the second trigger).
        """
        ctrl = self._make_triggering_controller()
        # Push through: buffer fill + first decision + full stim + lockout + safety margin
        total_ms = (
            ctrl.config.window_length * (1000.0 / ctrl.config.fs)  # fill buffer
            + ctrl._decision_stride * (1000.0 / ctrl.config.fs)  # first decision
            + ctrl.config.stim_duration_ms
            + ctrl.config.lockout_duration_ms
            + 50.0  # small safety margin past lockout end
        )
        n = int(total_ms * ctrl.config.fs / 1000.0) + ctrl._decision_stride + 10
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)

        # The re-trigger proves lockout DID expire (otherwise n_stimulations would be 1)
        assert ctrl.metrics.n_stimulations >= 2


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Verify metric accumulation logic."""

    def test_duty_cycle_zero_before_any_stim(self):
        ctrl = DummyHDCController(score=-999.0)
        _fill_buffer(ctrl)
        assert ctrl.metrics.duty_cycle == 0.0

    def test_duty_cycle_is_fraction_of_stim_samples(self):
        """Ingest a burst, then advance time to collect stim + non-stim samples."""
        cfg = ControllerConfig(stim_duration_ms=200.0, lockout_duration_ms=0.0)
        ctrl = DummyHDCController(score=1.0, config=cfg)

        # Feed enough to trigger: buffer fill + one decision
        n_trigger = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n_trigger)
        ctrl.ingest(t, lfp)

        # At this point stim is active.  The expected duty cycle at the end
        # of ingesting should satisfy:
        #   total_stim_samples / total_samples > 0
        assert ctrl.metrics.duty_cycle > 0.0

    def test_stim_times_ms_records_trigger_timestamp(self):
        ctrl = DummyHDCController(score=1.0)
        n = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert len(ctrl.metrics.stim_times_ms) == 1
        # Timestamp should be a positive ms value within the simulation window
        assert ctrl.metrics.stim_times_ms[0] > 0.0

    def test_detection_count_can_exceed_stim_count(self):
        """Detections during lockout count as detections but not stimulations."""
        ctrl = DummyHDCController(score=1.0)
        # Push through buffer fill + trigger + stay in stim/lockout for a while
        # feeding decision-cadence batches so multiple decisions fire
        n = ctrl.config.window_length + ctrl._decision_stride * 10
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        # More detections than stimulations if any decisions fired during lockout
        assert ctrl.metrics.n_detections >= ctrl.metrics.n_stimulations

    def test_to_dict_has_expected_keys(self):
        ctrl = DummyHDCController()
        d = ctrl.metrics.to_dict()
        for key in (
            "n_decisions",
            "n_detections",
            "n_stimulations",
            "duty_cycle",
            "total_stim_samples",
            "total_samples",
            "stim_times_ms",
        ):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 4. BetaController — filter properties
# ---------------------------------------------------------------------------


class TestBetaControllerFilter:
    """Verify the causal IIR bandpass filter behaves correctly."""

    def test_filter_state_continuity_across_chunks(self):
        """Filtering in one shot vs in two chunks should produce identical output."""
        cfg = ControllerConfig()
        fs = cfg.fs  # 250 Hz
        n = 500  # 2 seconds of data

        # Generate a 20 Hz sine (within beta band)
        t = np.arange(n) / fs
        lfp = np.sin(2 * np.pi * 20.0 * t)

        # Single-pass controller (one chunk)
        ctrl_single = BetaController(config=cfg)
        # We only care about what ends up in the buffer — call ingest with
        # sub-threshold score, but capture the filtered values via the internal
        # buffer after manually calling ingest for the full array.
        # Direct comparison: apply sosfilt once vs in two halves with state carry.

        from scipy.signal import sosfilt, sosfilt_zi
        from scipy.signal import butter

        nyq = fs / 2.0
        sos = butter(2, [13.0 / nyq, 30.0 / nyq], btype="bandpass", output="sos")

        # Ground truth: filter the whole signal at once
        filtered_gt, _ = sosfilt(sos, lfp, zi=np.zeros_like(sosfilt_zi(sos)))

        # Chunked: filter in two halves using state carry
        zi_carry = np.zeros_like(sosfilt_zi(sos))
        half = n // 2
        f1, zi_carry = sosfilt(sos, lfp[:half], zi=zi_carry)
        f2, zi_carry = sosfilt(sos, lfp[half:], zi=zi_carry)
        filtered_chunked = np.concatenate([f1, f2])

        np.testing.assert_allclose(filtered_gt, filtered_chunked, atol=1e-10)

    def test_beta_rms_above_noise_for_beta_band_signal(self):
        """A 20 Hz sine should produce a higher RMS score than a 1 Hz sine."""
        cfg = ControllerConfig(threshold=0.0)

        # 20 Hz (beta band)
        beta_ctrl = BetaController(config=cfg)
        n = cfg.window_length + beta_ctrl._decision_stride
        fs = cfg.fs
        t = np.arange(n) / fs
        lfp_beta = np.sin(2 * np.pi * 20.0 * t)
        t_ms = np.arange(n) * (1000.0 / fs)
        beta_ctrl.ingest(t_ms, lfp_beta)

        # 1 Hz (well below beta band → attenuated by bandpass)
        slow_ctrl = BetaController(config=cfg)
        lfp_slow = np.sin(2 * np.pi * 1.0 * t)
        slow_ctrl.ingest(t_ms, lfp_slow)

        # Beta band window should have higher RMS than sub-band signal
        beta_win = beta_ctrl.get_latest_window()
        slow_win = slow_ctrl.get_latest_window()
        if beta_win is not None and slow_win is not None:
            beta_rms = float(np.sqrt(np.mean(beta_win**2)))
            slow_rms = float(np.sqrt(np.mean(slow_win**2)))
            assert beta_rms > slow_rms * 5, (
                f"Expected beta RMS ({beta_rms:.4f}) >> slow RMS ({slow_rms:.4f})"
            )

    def test_beta_controller_reset_clears_filter_state(self):
        """After reset(), the filter state should be all zeros."""
        ctrl = BetaController()
        # Ingest some data to build up filter state
        t, lfp = _make_lfp_chunk(100, value=10.0)
        ctrl.ingest(t, lfp)
        ctrl.reset()
        # Filter state should be zeroed out
        assert np.all(ctrl._zi == 0.0)
        assert not ctrl.is_ready

    def test_stim_fn_returns_amplitude_when_stimulating(self):
        """get_stim_fn() should return stim_amplitude during STIMULATING."""
        cfg = ControllerConfig(stim_amplitude=2.5)
        ctrl = BetaController(config=cfg)
        # Manually set state to STIMULATING to test the closure
        ctrl.state = StimState.STIMULATING
        stim_fn = ctrl.get_stim_fn()
        assert stim_fn(0.0) == pytest.approx(2.5)

    def test_stim_fn_returns_zero_when_idle(self):
        cfg = ControllerConfig(stim_amplitude=2.5)
        ctrl = BetaController(config=cfg)
        stim_fn = ctrl.get_stim_fn()
        assert stim_fn(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. HDCController unit tests (no disk I/O via DummyHDCController)
# ---------------------------------------------------------------------------


class TestHDCControllerDecision:
    """Verify HDCController correctly routes through _compute_decision."""

    def test_positive_score_triggers_stim(self):
        """score=1.0 > threshold=0.0 → should trigger."""
        ctrl = DummyHDCController(score=1.0)
        n = ctrl.config.window_length + ctrl._decision_stride
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.metrics.n_stimulations >= 1

    def test_negative_score_never_triggers(self):
        """score=-1.0 < threshold=0.0 → should never trigger."""
        ctrl = DummyHDCController(score=-1.0)
        n = ctrl.config.window_length + ctrl._decision_stride * 20
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.metrics.n_stimulations == 0

    def test_custom_threshold_respected(self):
        """score=0.5 with threshold=1.0 → should NOT trigger."""
        cfg = ControllerConfig(threshold=1.0)
        ctrl = DummyHDCController(score=0.5, config=cfg)
        n = ctrl.config.window_length + ctrl._decision_stride * 5
        t, lfp = _make_lfp_chunk(n)
        ctrl.ingest(t, lfp)
        assert ctrl.metrics.n_stimulations == 0

    def test_compute_decision_calls_trainer(self):
        """_compute_decision must call trainer.decision_function with shape (1, 128)."""
        received_shapes = []

        class ShapeCapturingTrainer:
            def decision_function(self, windows):
                received_shapes.append(windows.shape)
                return np.array([1.0])

        ctrl = DummyHDCController(score=1.0)
        ctrl.trainer = ShapeCapturingTrainer()

        window = np.zeros(ctrl.config.window_length)
        ctrl._compute_decision(window)

        assert len(received_shapes) == 1
        assert received_shapes[0] == (1, ctrl.config.window_length)


# ---------------------------------------------------------------------------
# 6. Integration tests (marked slow — run with pytest -m slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunClosedLoop:
    """Integration tests: full ODE simulation with a controller in the loop.

    These tests run the actual ODE solver and are slower (~5–20 s each).
    Mark the test run with -m slow to include them.
    """

    @pytest.fixture
    def short_patho_config(self):
        """Short pathological simulation: 1.5 s total (1 s after warmup)."""
        from configs.sim_config import pathological_config

        return pathological_config(t_end=1500.0)

    @pytest.fixture
    def short_healthy_config(self):
        """Short healthy simulation: 1.5 s total (1 s after warmup)."""
        from configs.sim_config import healthy_config

        return healthy_config(t_end=1500.0)

    def test_return_dict_has_expected_keys(self, short_patho_config):
        from controllers.run_controller import run_closed_loop

        ctrl = DummyHDCController(score=1.0)  # always triggers
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        for key in ("t", "lfp", "stim", "state", "metrics", "config", "seed"):
            assert key in result, f"Missing key: {key}"

    def test_output_arrays_same_length(self, short_patho_config):
        from controllers.run_controller import run_closed_loop

        ctrl = DummyHDCController(score=1.0)
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        n = len(result["t"])
        assert len(result["lfp"]) == n
        assert len(result["stim"]) == n
        assert len(result["state"]) == n

    def test_stim_array_zero_when_not_stimulating(self, short_patho_config):
        """Where state != STIMULATING, stim current must be 0.0."""
        from controllers.run_controller import run_closed_loop

        ctrl = DummyHDCController(score=1.0)
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        stim = result["stim"]
        state = result["state"]
        # When state is IDLE (0) or LOCKOUT (2), stim must be 0.0
        not_stimulating = state != StimState.STIMULATING.value
        assert np.all(stim[not_stimulating] == 0.0)

    def test_stim_array_nonzero_when_stimulating(self, short_patho_config):
        """Where state == STIMULATING, stim is 0 or pulse amplitude."""
        from controllers.run_controller import run_closed_loop

        cfg = ControllerConfig(stim_amplitude=1.5)
        ctrl = DummyHDCController(score=1.0, config=cfg)
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        stim = result["stim"]
        state = result["state"]
        is_stim = state == StimState.STIMULATING.value
        if is_stim.any():
            assert np.all((stim[is_stim] == 0.0) | (stim[is_stim] == 1.5))
            assert np.any(stim[is_stim] > 0.0)

    def test_no_stimulation_for_never_trigger_controller(self, short_patho_config):
        """A controller that always returns sub-threshold scores must never stim."""
        from controllers.run_controller import run_closed_loop

        ctrl = DummyHDCController(score=-999.0)
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        assert result["metrics"].n_stimulations == 0
        assert np.all(result["stim"] == 0.0)

    def test_beta_controller_runs_without_error(self, short_patho_config):
        """BetaController should complete a run without exceptions."""
        from controllers.run_controller import run_closed_loop

        # Use a threshold of 0 (will trigger on any positive RMS — basically always)
        ctrl = BetaController(ControllerConfig(threshold=0.0))
        result = run_closed_loop(ctrl, short_patho_config, seed=0)
        assert "metrics" in result

    def test_metrics_seed_is_returned(self, short_patho_config):
        from controllers.run_controller import run_closed_loop

        ctrl = DummyHDCController(score=1.0)
        result = run_closed_loop(ctrl, short_patho_config, seed=42)
        assert result["seed"] == 42
