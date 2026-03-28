from __future__ import annotations

import unittest

import numpy as np

from hdc.search.config import SearchConfig
from hdc.search.run import make_dummy_validation_data
from hdc.search.validator import run_validator_search, top_candidates


class TestValidatorSearch(unittest.TestCase):
    def test_fixed_grid_has_expected_candidate_count(self) -> None:
        # Verifies default search grid expands to the expected number of encoder candidates.
        cfg = SearchConfig()
        specs = cfg.iter_encoder_specs()
        self.assertEqual(len(specs), 12)

    def test_search_returns_rows_for_all_readouts(self) -> None:
        # Verifies validator search evaluates each candidate with both readouts and reports finite metrics.
        cfg = SearchConfig(
            dimensions=(500,),
            bins=(8,),
            value_initializers=("random", "rff"),
            top_k=2,
            min_guardrail_auroc=0.50,
        )
        data = make_dummy_validation_data(seed=300, window_length=cfg.window_length)
        rows = run_validator_search(data=data, cfg=cfg)

        # 2 encoder candidates x 2 readouts
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.readout in {"prototype", "linear"} for row in rows))
        self.assertTrue(all(np.isfinite(row.balanced_accuracy_clean) for row in rows))
        self.assertTrue(all(np.isfinite(row.encoding_ms_per_window) for row in rows))

    def test_ranking_and_top_candidates(self) -> None:
        # Verifies ranking helper returns top-k winners with rank numbering applied.
        cfg = SearchConfig(
            dimensions=(500,),
            bins=(8,),
            value_initializers=("random",),
            top_k=1,
            min_guardrail_auroc=0.50,
        )
        data = make_dummy_validation_data(seed=301, window_length=cfg.window_length)
        rows = run_validator_search(data=data, cfg=cfg)

        winners = top_candidates(rows, cfg.top_k)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
