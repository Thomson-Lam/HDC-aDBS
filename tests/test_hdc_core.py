from __future__ import annotations

import unittest

import numpy as np

from hdc.dictionaries import DictionaryConfig, build_dictionaries
from hdc.encoder import EncoderConfig, WindowEncoder
from hdc.primitives import bind, bipolarize, bundle, normalized_dot
from hdc.readouts import LinearReadout, PrototypeReadout


class TestPrimitives(unittest.TestCase):
    def test_bind_and_bundle_are_bipolar(self) -> None:
        a = np.array([1, -1, 1, -1], dtype=np.int8)
        b = np.array([1, 1, -1, -1], dtype=np.int8)
        bound = bind(a, b)
        self.assertTrue(np.array_equal(bound, np.array([1, -1, -1, 1], dtype=np.int8)))

        stacked = np.stack([a, b, bound], axis=0)
        bundled = bundle(stacked, axis=0)
        self.assertEqual(set(np.unique(bundled).tolist()), {-1, 1})

    def test_bipolarize_tie_rule(self) -> None:
        x = np.array([-2.0, 0.0, 3.0])
        self.assertTrue(np.array_equal(bipolarize(x, zero_to=1), np.array([-1, 1, 1])))
        self.assertTrue(
            np.array_equal(bipolarize(x, zero_to=-1), np.array([-1, -1, 1]))
        )

    def test_normalized_dot_expected_values(self) -> None:
        v = np.array([1, -1, 1, -1], dtype=np.int8)
        self.assertAlmostEqual(float(normalized_dot(v, v)), 1.0, places=12)
        self.assertAlmostEqual(float(normalized_dot(v, -v)), -1.0, places=12)


class TestDictionariesAndEncoder(unittest.TestCase):
    def test_dictionary_determinism(self) -> None:
        cfg = DictionaryConfig(
            dimension=1000,
            n_bins=8,
            window_length=128,
            value_init="random",
            seed=123,
        )
        v1, p1 = build_dictionaries(cfg)
        v2, p2 = build_dictionaries(cfg)

        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(np.array_equal(p1, p2))
        self.assertEqual(v1.shape, (8, 1000))
        self.assertEqual(p1.shape, (128, 1000))

    def test_encoder_output_shape_and_domain(self) -> None:
        encoder = WindowEncoder(
            EncoderConfig(
                dimension=1000,
                n_bins=16,
                window_length=128,
                value_init="rff",
                seed=7,
            )
        )
        rng = np.random.default_rng(0)
        windows = rng.standard_normal(size=(10, 128))
        hvs = encoder.encode_batch(windows)

        self.assertEqual(hvs.shape, (10, 1000))
        self.assertEqual(hvs.dtype, np.int8)
        self.assertEqual(set(np.unique(hvs).tolist()), {-1, 1})

    def test_constant_window_is_handled(self) -> None:
        encoder = WindowEncoder(
            EncoderConfig(dimension=500, n_bins=8, window_length=128, seed=5)
        )
        w = np.ones(128, dtype=np.float64)
        hv = encoder.encode_window(w)
        self.assertEqual(hv.shape, (500,))
        self.assertEqual(set(np.unique(hv).tolist()), {-1, 1})


class TestReadouts(unittest.TestCase):
    def test_readout_interfaces_and_sign_behavior(self) -> None:
        rng = np.random.default_rng(42)
        d = 1000
        n = 50

        base = rng.choice(np.array([-1, 1], dtype=np.int8), size=d)

        healthy = np.tile(base, (n, 1))
        pathological = np.tile(-base, (n, 1))

        healthy[:, :60] *= -1
        pathological[:, 60:120] *= -1

        x = np.vstack([healthy, pathological]).astype(np.int8)
        y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

        proto = PrototypeReadout().fit(x, y)
        lin = LinearReadout(seed=123).fit(x, y)

        proto_scores = proto.decision_function(x)
        lin_scores = lin.decision_function(x)

        self.assertGreater(float(np.mean(proto_scores[y == 1])), 0.0)
        self.assertLess(float(np.mean(proto_scores[y == 0])), 0.0)
        self.assertGreater(float(np.mean(lin_scores[y == 1])), 0.0)
        self.assertLess(float(np.mean(lin_scores[y == 0])), 0.0)

        proto_pred = proto.predict(x)
        lin_pred = lin.predict(x)

        self.assertEqual(proto_pred.shape, (2 * n,))
        self.assertEqual(lin_pred.shape, (2 * n,))


if __name__ == "__main__":
    unittest.main()
