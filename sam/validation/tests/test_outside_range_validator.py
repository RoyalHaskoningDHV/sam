import unittest

import pandas as pd
import numpy as np
from sam.validation import OutsideRangeValidator
from pandas.testing import assert_frame_equal


class TestOutsideRangeValidator(unittest.TestCase):
    X_train = pd.DataFrame(
        {
            "A": [1, 2, 6, 3, 4, 4, 4],
            "B": [2, 3, 4, 5, 9, 4, 2],
        }
    )

    X_test = pd.DataFrame(
        {
            "A": [0, 7, 3, 4, 5],
            "B": [10, 0, 1, 2, 3],
        }
    )

    def test_no_outliers(self):
        RF = OutsideRangeValidator(min_value=0, max_value=10)
        data_corrected = RF.fit_transform(self.X_train)
        assert_frame_equal(data_corrected, self.X_train)

    def test_all_above_max(self):
        RF = OutsideRangeValidator(max_value=0)
        expected = pd.DataFrame(
            {
                "A": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        data_corrected = RF.fit_transform(self.X_train)
        assert_frame_equal(data_corrected, expected)

    def test_all_below_min(self):
        RF = OutsideRangeValidator(min_value=10)
        expected = pd.DataFrame(
            {
                "A": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        data_corrected = RF.fit_transform(self.X_train)
        assert_frame_equal(data_corrected, expected)

    def test_auto(self):
        RF = OutsideRangeValidator(min_value="auto", max_value="auto")
        data_corrected = RF.fit_transform(self.X_train)
        assert_frame_equal(data_corrected, self.X_train)
        data_corrected_test = RF.transform(self.X_test)
        expected_test = pd.DataFrame(
            {
                "A": [np.nan, np.nan, 3, 4, 5],
                "B": [np.nan, np.nan, np.nan, 2, 3],
            }
        )
        assert_frame_equal(data_corrected_test, expected_test)

    def test_single_col_auto(self):
        RF = OutsideRangeValidator(cols=["A"], min_value="auto", max_value="auto")
        data_corrected = RF.fit_transform(self.X_train)
        assert_frame_equal(data_corrected, self.X_train)
        data_corrected_test = RF.transform(self.X_test)
        expected_test = pd.DataFrame(
            {
                "A": [np.nan, np.nan, 3, 4, 5],
                "B": [10, 0, 1, 2, 3],
            }
        )
        assert_frame_equal(data_corrected_test, expected_test)
