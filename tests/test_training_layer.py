from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hdc.encoder import EncoderConfig
from hdc.search.run import make_dummy_validation_data
from hdc.training import LinearHDCTrainer, PrototypeHDCTrainer


class TestTrainingLayer(unittest.TestCase):
    def test_prototype_trainer_fit_predict_and_reload(self) -> None:
        # Verifies prototype trainer can fit/evaluate, persist, reload, and preserve decision scores.
        data = make_dummy_validation_data(seed=909, window_length=128)
        cfg = EncoderConfig(
            dimension=1000,
            n_bins=16,
            window_length=128,
            value_init="rff",
            seed=44,
        )

        trainer = PrototypeHDCTrainer(cfg).fit(data.train.x, data.train.y)
        metrics = trainer.evaluate(data.val_clean.x, data.val_clean.y)
        self.assertGreater(metrics["balanced_accuracy"], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prototype"
            trainer.save(path)
            loaded = PrototypeHDCTrainer.load(path)

            s1 = trainer.decision_function(data.val_clean.x)
            s2 = loaded.decision_function(data.val_clean.x)
            self.assertTrue(np.allclose(s1, s2))

    def test_linear_trainer_fit_predict_and_reload(self) -> None:
        # Verifies linear trainer can fit/evaluate, persist, reload, and preserve decision scores.
        data = make_dummy_validation_data(seed=910, window_length=128)
        cfg = EncoderConfig(
            dimension=1000,
            n_bins=8,
            window_length=128,
            value_init="random",
            seed=55,
        )

        trainer = LinearHDCTrainer(cfg, seed=12).fit(data.train.x, data.train.y)
        metrics = trainer.evaluate(data.val_clean.x, data.val_clean.y)
        self.assertGreater(metrics["balanced_accuracy"], 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "linear"
            trainer.save(path)
            loaded = LinearHDCTrainer.load(path)

            s1 = trainer.decision_function(data.val_clean.x)
            s2 = loaded.decision_function(data.val_clean.x)
            self.assertTrue(np.allclose(s1, s2))


if __name__ == "__main__":
    unittest.main()
