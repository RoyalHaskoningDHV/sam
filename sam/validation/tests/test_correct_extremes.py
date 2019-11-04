import unittest
from sam.validation import RemoveExtremeValues
from .numeric_assertions import NumericAssertions
import pandas as pd
import numpy as np


class NumericAssertions:
    """
    This class is following the UnitTest naming conventions.
    It is meant to be used along with unittest.TestCase like so:

    class MyTest(unittest.TestCase, NumericAssertions):
        ...
    It needs python >= 2.6
    """

    def assertAllNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """
        standardMsg = "Not all values are NaN"
        try:
            if not np.all(np.isnan(value)):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            self.fail(self._formatMessage(msg, standardMsg))

    def assertAllNotNaN(self, value, msg=None):
        """
        Fail if provided value is NaN
        """
        standardMsg = "There is at least 1 NaN in provided series"
        try:
            if np.all(~np.isnan(value)):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            pass


class TestRemoveExtremes(unittest.TestCase, NumericAssertions):

    def test_remove_extreme_values(self):

        # create some random data
        np.random.seed(10)
        test_data = np.random.random(size=(100))
        # with one clear outlier
        test_data[25] *= 15
        test_df = pd.DataFrame()
        test_df['values'] = test_data
        # now detect extremes
        cols_to_check = ['values']
        REV = RemoveExtremeValues(
            cols=cols_to_check,
            rollingwindow=10,
            madthresh=10)
        data_corrected = REV.fit_transform(test_df)

        self.assertAllNaN(data_corrected.iloc[25])
        self.assertAllNotNaN(data_corrected.drop([25], axis=0))
