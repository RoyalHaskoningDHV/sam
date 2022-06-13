import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sam.preprocessing import inverse_differenced_target, make_differenced_target


class TestDifferencing(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"X": [18, 19, 20, 21], "y": [100, 200, 500, 1000]})

        self.expected = pd.DataFrame(
            {
                "y_diff_1": [100, 300, 500, np.nan],
                "y_diff_2": [400, 800, np.nan, np.nan],
                "y_diff_3": [900, np.nan, np.nan, np.nan],
            }
        )

        time1 = "2019/03/11 00:00:00"
        time2 = "2019/03/11 03:00:00"
        freq = "1h"
        self.daterange = pd.date_range(time1, time2, freq=freq)

    def test_create_target(self):
        result = make_differenced_target(self.df["y"], lags=[1, 2, 3])
        assert_frame_equal(result, self.expected)

    def test_invert_target(self):
        result = inverse_differenced_target(self.expected, self.df["y"])

        expected = pd.DataFrame(
            {
                "y_diff_1": [200, 500, 1000, np.nan],
                "y_diff_2": [500, 1000, np.nan, np.nan],
                "y_diff_3": [1000, np.nan, np.nan, np.nan],
            }
        )
        assert_frame_equal(result, expected)

    def test_prefix_col(self):

        result = make_differenced_target(self.df["y"], lags=[1, 2, 3], newcol_prefix="mycol")
        expected = self.expected.copy()
        expected.columns = ["mycol_diff_1", "mycol_diff_2", "mycol_diff_3"]
        assert_frame_equal(result, expected)

    def test_negative_lag(self):
        self.assertRaises(ValueError, make_differenced_target, self.df["y"], lags=[1, 2, -3])

    def test_fractional_lag(self):
        self.assertRaises(ValueError, make_differenced_target, self.df["y"], lags=[1, 2, 2.5])

    def test_series_output(self):
        result = make_differenced_target(self.df["y"], lags=2)
        expected = self.expected["y_diff_2"]
        assert_series_equal(result, expected)
