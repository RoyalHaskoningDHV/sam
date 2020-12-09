import unittest
from numpy.testing import assert_array_equal
# Below are needed for setting up tests
from sam.metrics import compute_quantile_ratios
import pandas as pd
import numpy as np


class TestQuantileRatios(unittest.TestCase):

    def test_quantile_ratios(self):

        y = pd.Series([1, 1.5, 1, 2, 5])
        y_pred = pd.DataFrame({
            'predict_lead_0_q_0.2': [1, 2, 1, 2, 1],
            'predict_lead_0_q_0.8': [3, 4, 3, 4, 3]})

        res = compute_quantile_ratios(y, y_pred)

        assert_array_equal(list(res.values()), [0.2, 0.8])
        assert_array_equal(list(res.keys()), [0.2, 0.8])


if __name__ == '__main__':
    unittest.main()
