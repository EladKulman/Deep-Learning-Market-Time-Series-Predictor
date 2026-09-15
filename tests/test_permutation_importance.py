import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from permutation_importance import draw_donors, swap_feature, window_losses  # noqa: E402


class SwapTests(unittest.TestCase):
    def test_donors_never_self(self):
        rng = np.random.default_rng(0)
        for n in (2, 3, 7, 50):
            donors = draw_donors(rng, n)
            self.assertEqual(sorted(donors.tolist()), list(range(n)))
            self.assertFalse(np.any(donors == np.arange(n)))

    def test_swap_replaces_only_target_feature_and_keeps_originals(self):
        inputs = []
        for i in range(3):
            n = 10 + i
            inputs.append({
                "target": np.full(n, float(i), dtype="float32"),
                "past_covariates": {"a": np.full(n, 10.0 * i, dtype="float32"), "k": np.full(n, 100.0 * i, dtype="float32")},
                "future_covariates": {"k": np.full(2, 1000.0 * i, dtype="float32")},
            })
        donors = np.array([1, 2, 0])
        swapped = swap_feature(inputs, "a", False, donors)
        self.assertTrue(np.all(swapped[0]["past_covariates"]["a"] == 10.0))       # from window 1 (tail-aligned)
        self.assertEqual(len(swapped[0]["past_covariates"]["a"]), 10)
        self.assertTrue(np.all(swapped[0]["past_covariates"]["k"] == 0.0))        # untouched
        self.assertTrue(np.all(inputs[0]["past_covariates"]["a"] == 0.0))         # originals intact
        # shorter donor gets NaN left-padding
        self.assertTrue(np.isnan(swapped[2]["past_covariates"]["a"][:2]).all())
        self.assertTrue(np.all(swapped[2]["past_covariates"]["a"][2:] == 0.0))
        known = swap_feature(inputs, "k", True, donors)
        self.assertTrue(np.all(known[0]["future_covariates"]["k"] == 1000.0))
        self.assertTrue(np.all(inputs[0]["future_covariates"]["k"] == 0.0))

    def test_window_losses_matches_wql_definition(self):
        levels = [0.1, 0.5, 0.9]
        rows = []
        for h, actual in enumerate([0.01, -0.02], start=1):
            rows.append({"window": 0, "horizon": h, "actual": actual, "q0.1": -0.01, "q0.5": 0.0, "q0.9": 0.01})
        losses = window_losses(pd.DataFrame(rows), levels, 2)
        actual = np.array([[0.01], [-0.02]]); preds = np.array([[-0.01, 0.0, 0.01]] * 2); q = np.array([levels])
        err = actual - preds
        expected = 2 * np.maximum(q * err, (q - 1) * err).mean() / np.abs(actual).mean()
        self.assertAlmostEqual(losses.loc[0, "wql"], expected, places=10)
        self.assertEqual(set(losses.columns), {"wql", "denom", "wql_h1", "wql_h2"})


if __name__ == "__main__":
    unittest.main()
