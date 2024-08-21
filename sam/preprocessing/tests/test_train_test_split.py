import unittest

import pandas as pd
from sam.preprocessing import datetime_train_test_split
from pandas.testing import assert_frame_equal, assert_series_equal


class TestTrainTestSplit(unittest.TestCase):
    data = pd.DataFrame(
        {
            "TIME": pd.date_range("2022-01-01", periods=10, freq="h", tz=None),
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        }
    )
    data = data.set_index("TIME")
    split_date = "2022-01-01 07:00:00"

    def test_single_split(self):
        train, test = datetime_train_test_split(self.data, datetime=self.split_date)
        assert_frame_equal(train, self.data.iloc[:7])
        assert_frame_equal(test, self.data.iloc[7:])
        self.assertEqual(train.shape, (7, 2))
        self.assertEqual(test.shape, (3, 2))

    def test_multiple_splits(self):
        data_a, data_b = self.data[["a"]], self.data["b"]
        train_a, test_a, train_b, test_b = datetime_train_test_split(
            data_a, data_b, datetime=self.split_date
        )

        assert_frame_equal(train_a, data_a.iloc[:7])
        assert_frame_equal(test_a, data_a.iloc[7:])
        assert_series_equal(train_b, data_b.iloc[:7])
        assert_series_equal(test_b, data_b.iloc[7:])
        self.assertEqual(train_a.shape, (7, 1))
        self.assertEqual(test_a.shape, (3, 1))
        self.assertEqual(train_b.shape, (7,))
        self.assertEqual(test_b.shape, (3,))

    def test_datetimecol(self):
        data = self.data.reset_index()
        train, test = datetime_train_test_split(data, datetime=self.split_date, datecol="TIME")
        assert_frame_equal(train, data.iloc[:7])
        assert_frame_equal(test, data.iloc[7:])
        self.assertEqual(train.shape, (7, 3))
        self.assertEqual(test.shape, (3, 3))
