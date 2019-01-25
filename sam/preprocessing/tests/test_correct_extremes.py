import unittest
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_equal

from sam.preprocessing import correct_below_threshold
from sam.preprocessing import correct_above_threshold
from sam.preprocessing import correct_outside_range
import pandas as pd
import numpy as np


class TestCorrectExtremes(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                               "TARGET": [0, 2, -1, 2, 2, 1]
                               }, columns=["TEST", "TARGET"])

    def testDefaultAboveThreshold(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [0, np.nan, -1, np.nan, np.nan, 1]
                                 }, columns=["TEST", "TARGET"])

        result = correct_above_threshold(self.X)
        assert_frame_equal(result, expected)

    def testRemoveInRange(self):
        expected = pd.DataFrame({"TEST": [1, 7],
                                 "TARGET": [0, 1]
                                 }, columns=["TEST", "TARGET"])

        # Indexes are left intact when removing
        expected.index = pd.RangeIndex(0, 6, 5)

        result = correct_outside_range(self.X, method='remove')
        assert_frame_equal(result, expected, check_dtype=False)

    def testSetMaxValue(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [0, 99, -1, 99, 99, 1]
                                 }, columns=["TEST", "TARGET"])

        result = correct_above_threshold(self.X,
                                         method="value",
                                         value=99)
        assert_frame_equal(result, expected, check_dtype=False)

    def testInterpolateTime(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [0.0, -0.5, -1.0, -0.5, 0.0, 1.0]
                                 }, columns=["TEST", "TARGET"])
        expected = expected.set_index(pd.DatetimeIndex(expected['TEST']))

        time_index_df = self.X.set_index(pd.DatetimeIndex(self.X['TEST']))
        result = correct_above_threshold(time_index_df, method="average")
        assert_frame_equal(result, expected)

    def testForwardFill(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [0, 0, -1, -1, -1, 1]
                                 }, columns=["TEST", "TARGET"])

        result = correct_above_threshold(self.X, method="previous")
        assert_frame_equal(result, expected, check_dtype=False)

    def testCutOffTuple(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [0, 1, 0, 1, 1, 1]
                                 }, columns=["TEST", "TARGET"])

        result = correct_outside_range(self.X, method="clip")
        assert_frame_equal(result, expected, check_dtype=False)

    def testCutOffBelow(self):
        expected = pd.DataFrame({"TEST": [1, 2, 3, 4, 5, 7],
                                 "TARGET": [1, 2, 1, 2, 2, 1]
                                 }, columns=["TEST", "TARGET"])

        result = correct_below_threshold(self.X, method="clip", threshold=1)
        assert_frame_equal(result, expected, check_dtype=False)

    def testIncorrectMethod(self):
        self.assertRaises(AssertionError, correct_below_threshold, self.X, method="error")
