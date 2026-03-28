from __future__ import annotations

import unittest

import numpy as np

from hdc.encoder import EncoderConfig, WindowEncoder
from hdc.readouts import LinearReadout, PrototypeReadout


def _make_dummy_windows(
    seed: int, n: int, window_length: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(window_length, dtype=np.float64) / 250.0
    beta_carrier = np.sin(2.0 * np.pi * 20.0 * t)

    healthy = 0.8 * rng.standard_normal(size=(n, window_length))
    pathological = (
        0.8 * rng.standard_normal(size=(n, window_length)) + 1.1 * beta_carrier
    )
    return healthy, pathological


class TestDummyReproducibility(unittest.TestCase):
    def test_encoder_same_seed_same_hypervectors_random_init(self) -> None:
        healthy, pathological = _make_dummy_windows(seed=21, n=32, window_length=128)
        windows = np.vstack([healthy, pathological])

        cfg = EncoderConfig(
            dimension=1000,
            n_bins=16,
            window_length=128,
            value_init="random",
            seed=99,
        )
        enc_a = WindowEncoder(cfg)
        enc_b = WindowEncoder(cfg)

        hv_a = enc_a.encode_batch(windows)
        hv_b = enc_b.encode_batch(windows)

        self.assertTrue(np.array_equal(hv_a, hv_b))

    def test_encoder_same_seed_same_hypervectors_rff_init(self) -> None:
        healthy, pathological = _make_dummy_windows(seed=22, n=32, window_length=128)
        windows = np.vstack([healthy, pathological])

        cfg = EncoderConfig(
            dimension=1000,
            n_bins=8,
            window_length=128,
            value_init="rff",
            seed=123,
        )
        enc_a = WindowEncoder(cfg)
        enc_b = WindowEncoder(cfg)

        hv_a = enc_a.encode_batch(windows)
        hv_b = enc_b.encode_batch(windows)

        self.assertTrue(np.array_equal(hv_a, hv_b))

    def test_encoder_different_seed_changes_hypervectors(self) -> None:
        healthy, pathological = _make_dummy_windows(seed=23, n=20, window_length=128)
        windows = np.vstack([healthy, pathological])

        enc_a = WindowEncoder(
            EncoderConfig(
                dimension=1000,
                n_bins=16,
                window_length=128,
                value_init="random",
                seed=10,
            )
        )
        enc_b = WindowEncoder(
            EncoderConfig(
                dimension=1000,
                n_bins=16,
                window_length=128,
                value_init="random",
                seed=11,
            )
        )

        hv_a = enc_a.encode_batch(windows)
        hv_b = enc_b.encode_batch(windows)

        self.assertFalse(np.array_equal(hv_a, hv_b))

    def test_readout_reproducibility_with_fixed_seed(self) -> None:
        healthy, pathological = _make_dummy_windows(seed=30, n=60, window_length=128)
        windows = np.vstack([healthy, pathological])
        y = np.concatenate(
            [
                np.zeros(healthy.shape[0], dtype=np.int64),
                np.ones(pathological.shape[0], dtype=np.int64),
            ]
        )

        encoder = WindowEncoder(
            EncoderConfig(
                dimension=2000,
                n_bins=16,
                window_length=128,
                value_init="rff",
                seed=7,
            )
        )
        x = encoder.encode_batch(windows)

        proto_a = PrototypeReadout().fit(x, y)
        proto_b = PrototypeReadout().fit(x, y)
        self.assertTrue(
            np.array_equal(proto_a.healthy_prototype, proto_b.healthy_prototype)
        )
        self.assertTrue(np.array_equal(proto_a.path_prototype, proto_b.path_prototype))

        lin_a = LinearReadout(seed=5).fit(x, y)
        lin_b = LinearReadout(seed=5).fit(x, y)

        self.assertTrue(np.allclose(lin_a.model.coef_, lin_b.model.coef_))
        self.assertTrue(np.allclose(lin_a.model.intercept_, lin_b.model.intercept_))

        scores_a = lin_a.decision_function(x)
        scores_b = lin_b.decision_function(x)
        self.assertTrue(np.allclose(scores_a, scores_b))

    def test_end_to_end_scores_reproducible(self) -> None:
        healthy_train, path_train = _make_dummy_windows(
            seed=100, n=80, window_length=128
        )
        healthy_val, path_val = _make_dummy_windows(seed=101, n=40, window_length=128)

        x_train_raw = np.vstack([healthy_train, path_train])
        y_train = np.concatenate(
            [
                np.zeros(healthy_train.shape[0], dtype=np.int64),
                np.ones(path_train.shape[0], dtype=np.int64),
            ]
        )
        x_val_raw = np.vstack([healthy_val, path_val])

        cfg = EncoderConfig(
            dimension=5000,
            n_bins=16,
            window_length=128,
            value_init="rff",
            seed=44,
        )

        enc_a = WindowEncoder(cfg)
        enc_b = WindowEncoder(cfg)

        x_train_a = enc_a.encode_batch(x_train_raw)
        x_train_b = enc_b.encode_batch(x_train_raw)
        x_val_a = enc_a.encode_batch(x_val_raw)
        x_val_b = enc_b.encode_batch(x_val_raw)

        self.assertTrue(np.array_equal(x_train_a, x_train_b))
        self.assertTrue(np.array_equal(x_val_a, x_val_b))

        proto_a = PrototypeReadout().fit(x_train_a, y_train)
        proto_b = PrototypeReadout().fit(x_train_b, y_train)
        proto_scores_a = proto_a.decision_function(x_val_a)
        proto_scores_b = proto_b.decision_function(x_val_b)
        self.assertTrue(np.allclose(proto_scores_a, proto_scores_b))

        lin_a = LinearReadout(seed=12).fit(x_train_a, y_train)
        lin_b = LinearReadout(seed=12).fit(x_train_b, y_train)
        lin_scores_a = lin_a.decision_function(x_val_a)
        lin_scores_b = lin_b.decision_function(x_val_b)
        self.assertTrue(np.allclose(lin_scores_a, lin_scores_b))


if __name__ == "__main__":
    unittest.main()
