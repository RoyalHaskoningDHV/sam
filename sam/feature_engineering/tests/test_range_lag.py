import unittest

import pandas as pd
from pandas.testing import assert_series_equal
from sam.feature_engineering import range_lag_column


class TestRangeLagColumn(unittest.TestCase):
    def test_positive_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([1, 1, 0, 0, 1, 1, 0])
        assert_series_equal(range_lag_column(testserie, (1, 2)), lagserie, check_dtype=False)

    def test_negative_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([0, 0, 1, 1, 0, 0, 1])
        assert_series_equal(range_lag_column(testserie, (-1, 0)), lagserie, check_dtype=False)

    def test_wrong_sorted_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([0, 1, 1, 1, 0, 1, 1])
        assert_series_equal(range_lag_column(testserie, (1, -1)), lagserie, check_dtype=False)

    def test_lag_with_floats_negative(self):
        testserie = [0.4, 0.4, 0.1, 0.2, 0.6, 0.5, 0.1]
        expected = pd.Series([0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6])
        assert_series_equal(range_lag_column(testserie, (-2, 0)), expected)

    def test_lag_with_floats_positive(self):
        testserie = [0.4, 0.4, 0.1, 0.2, 0.6, 0.5, 0.1]
        expected = pd.Series([0.4, 0.2, 0.6, 0.6, 0.5, 0.1, 0.0])
        assert_series_equal(range_lag_column(testserie, (1, 2)), expected)

    def test_lag_with_floats_bothsides(self):
        testserie = [0.4, 0.4, 0.1, 0.2, 0.6, 0.5, 0.1]
        expected = pd.Series([0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.5])
        assert_series_equal(range_lag_column(testserie, (-1, 1)), expected)

    def test_duplicate_axis(self):
        testserie = pd.Series([0, 1, 0, 1, 0], index=[1, 1, 1, 1, 1])
        expected = pd.Series([1, 1, 1, 1, 0], index=[1, 1, 1, 1, 1])
        assert_series_equal(range_lag_column(testserie, (0, 1)), expected, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
