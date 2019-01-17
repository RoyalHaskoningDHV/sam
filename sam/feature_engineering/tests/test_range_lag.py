import unittest
from pandas.testing import assert_series_equal

import pandas as pd
from sam.feature_engineering import range_lag_column


class TestRangeLagColumn(unittest.TestCase):

    def test_positive_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([1, 1, 0, 0, 1, 1, 0])
        assert_series_equal(range_lag_column(testserie, (1, 2)), lagserie)

    def test_negative_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([0, 0, 1, 1, 0, 0, 1])
        assert_series_equal(range_lag_column(testserie, (-1, 0)), lagserie)

    def test_wrong_sorted_lag(self):
        testserie = pd.Series([0, 0, 1, 0, 0, 0, 1])
        lagserie = pd.Series([0, 1, 1, 1, 0, 1, 1])
        assert_series_equal(range_lag_column(testserie, (1, -1)), lagserie)

if __name__ == '__main__':
    unittest.main()
