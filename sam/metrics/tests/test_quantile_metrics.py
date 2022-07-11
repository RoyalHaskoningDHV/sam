import unittest

import pandas as pd
from numpy.testing import assert_array_equal
from sam.metrics import compute_quantile_crossings, compute_quantile_ratios


class TestQuantileMetrics(unittest.TestCase):
    def test_quantile_ratios(self):

        y = pd.Series([1, 1.5, 1, 2, 5])
        pred = pd.DataFrame(
            {
                "predict_lead_0_q_0.2": [1, 2, 1, 2, 1],
                "predict_lead_0_q_0.8": [3, 4, 3, 4, 3],
            }
        )

        res = compute_quantile_ratios(y, pred)

        assert_array_equal(list(res.values()), [0.2, 0.8])
        assert_array_equal(list(res.keys()), [0.2, 0.8])

    def test_quantile_crossings(self):

        pred = pd.DataFrame(
            {
                "predict_lead_0_q_0.2": [1, 3, 5, 4],
                "predict_lead_0_q_0.4": [6, 2, 4, 5],
                "predict_lead_0_mean": [3, 1, 3, 6],
                "predict_lead_0_q_0.6": [4, 5, 7, 7],
                "predict_lead_0_q_0.8": [5, 6, 6, 8],
            }
        )

        res = compute_quantile_crossings(pred)
        expected_keys = [
            "0.800 < 0.600",
            "0.600 < mean",
            "mean < 0.400",
            "0.400 < 0.200",
        ]
        expected_values = [0.25, 0.0, 0.75, 0.5]
        assert_array_equal(list(res.keys()), expected_keys)
        assert_array_equal(list(res.values()), expected_values)

        res = compute_quantile_crossings(pred, qs=[0.6, 0.4])
        expected_keys = ["0.600 < 0.400"]
        expected_values = [0.25]
        assert_array_equal(list(res.keys()), expected_keys)
        assert_array_equal(list(res.values()), expected_values)

        res = compute_quantile_crossings(pred, qs=[0.8, "mean"])
        expected_keys = ["0.800 < mean"]
        expected_values = [0.0]
        assert_array_equal(list(res.keys()), expected_keys)
        assert_array_equal(list(res.values()), expected_values)

        self.assertRaises(ValueError, compute_quantile_crossings, pred, 0, [0.5, "mean"])

        pred = pd.DataFrame(
            {"predict_lead_0_q_0.5": [6, 2, 4, 5], "predict_lead_0_mean": [3, 1, 3, 6]}
        )

        self.assertRaises(ValueError, compute_quantile_crossings, pred)


if __name__ == "__main__":
    unittest.main()
