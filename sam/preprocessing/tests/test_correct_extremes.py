import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sam.preprocessing import (
    correct_above_threshold,
    correct_below_threshold,
    correct_outside_range,
)


class TestCorrectExtremes(unittest.TestCase):
    def setUp(self):
        self.X = pd.Series([0, 2, -1, 2, 2, 1])

    def testDefaultAboveThreshold(self):
        expected = pd.Series([0, np.nan, -1, np.nan, np.nan, 1])
        result = correct_above_threshold(self.X)
        assert_series_equal(result, expected)

    def testRemoveInRange(self):
        expected = pd.Series([0, 1], index=[0, 5])
        result = correct_outside_range(self.X, method="remove")
        assert_series_equal(result, expected, check_dtype=False)

    def testSetMaxValue(self):
        expected = pd.Series([0, 99, -1, 99, 99, 1])
        result = correct_above_threshold(self.X, method="value", value=99)
        assert_series_equal(result, expected, check_dtype=False)

    def testInterpolateTime(self):
        expected = pd.Series([0.0, -0.5, -1.0, -0.5, 0.0, 1.0])
        index = pd.DatetimeIndex(pd.Series([1, 2, 3, 4, 5, 7]))

        expected.index = index
        time_series = self.X.copy()
        time_series.index = index

        result = correct_above_threshold(time_series, method="average")
        assert_series_equal(result, expected)

    def testForwardFill(self):
        expected = pd.Series([0, 0, -1, -1, -1, 1])
        result = correct_above_threshold(self.X, method="previous")
        assert_series_equal(result, expected, check_dtype=False)

    def testCutOffTuple(self):
        expected = pd.Series([0, 1, 0, 1, 1, 1])
        result = correct_outside_range(self.X, method="clip")
        assert_series_equal(result, expected, check_dtype=False)

    def testCutOffBelow(self):
        expected = pd.Series([1, 2, 1, 2, 2, 1])
        result = correct_below_threshold(self.X, method="clip", threshold=1)
        assert_series_equal(result, expected, check_dtype=False)

    def testIncorrectMethod(self):
        self.assertRaises(ValueError, correct_below_threshold, self.X, method="error")

    def testNanChange(self):
        # Test if nans are ignored as they should be
        X = pd.Series([0, np.nan, -1, 2, 2, 1])
        result = correct_below_threshold(X, method="previous", threshold=0)
        expected = pd.Series([0, np.nan, 0, 2, 2, 1])
        assert_series_equal(result, expected)
