"""
HDC closed-loop controller.

Loads a trained HDCTrainer model from disk and uses its decision_function()
to score each incoming LFP window.  The base class handles all buffering,
state machine transitions, and metric accumulation.

Decision pipeline (per window):
    raw LFP window (128 samples, mV)
        → trainer.decision_function(window[None, :])
            → WindowEncoder.zscore_window()        (per-window z-score, inside encoder)
            → quantize() → bind() → bundle()       (HDC encoding)
            → readout.decision_function()           (prototype diff or logistic margin)
        → scalar margin score (positive = pathological, negative = healthy)
        → compared against config.threshold

Note: z-score normalisation is applied inside WindowEncoder, so raw mV values
are passed here without any pre-processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseController, ControllerConfig
from hdc.training import BaseHDCTrainer, LinearHDCTrainer, PrototypeHDCTrainer


class HDCController(BaseController):
    """Closed-loop DBS controller backed by a trained HDC classifier.

    Parameters
    ----------
    model_path   : path to a directory produced by BaseHDCTrainer.save().
                   Must contain metadata.json, encoder_dictionaries.npz,
                   and readout.npz.
    config       : ControllerConfig — control loop timing and thresholds.
                   Defaults to experiment-spec values if not provided.
    trainer_type : "linear" or "prototype" — must match the type of model
                   that was saved at model_path.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: ControllerConfig | None = None,
        trainer_type: str = "linear",
    ) -> None:
        super().__init__(config or ControllerConfig())
        self.model_path   = Path(model_path)
        self.trainer_type = trainer_type
        # Load the frozen model.  The trainer carries the encoder config,
        # value/position dictionaries, and readout weights.
        self.trainer: BaseHDCTrainer = self._load_trainer(trainer_type)

    def _load_trainer(self, trainer_type: str) -> BaseHDCTrainer:
        """Instantiate the correct trainer class and load weights from disk."""
        if trainer_type == "linear":
            return LinearHDCTrainer.load(self.model_path)
        elif trainer_type == "prototype":
            return PrototypeHDCTrainer.load(self.model_path)
        else:
            raise ValueError(
                f"Unknown trainer_type {trainer_type!r}. "
                "Expected 'linear' or 'prototype'."
            )

    def _compute_decision(self, window: np.ndarray) -> float:
        """Encode a single LFP window and return the HDC margin score.

        Parameters
        ----------
        window : (window_length,) float64 — raw LFP samples in mV,
                 chronologically ordered (oldest first).

        Returns
        -------
        float — margin score from the readout's decision_function.
                Positive → pathological side of the decision boundary.
                Negative → healthy side.
                0.0 is the natural threshold (config.threshold default).

        Implementation note
        -------------------
        decision_function expects shape (n_windows, window_length), so we
        expand dims with [None, :] to create a batch of one window.
        The result is a (1,) array; we index [0] to unwrap the scalar.
        """
        # Expand to (1, window_length) for the batched encoder interface
        scores = self.trainer.decision_function(window[None, :])
        return float(scores[0])
