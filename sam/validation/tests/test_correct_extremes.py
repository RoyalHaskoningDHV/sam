import unittest

import numpy as np
import pandas as pd
from sam.validation import RemoveExtremeValues

from .numeric_assertions import NumericAssertions


class TestRemoveExtremes(unittest.TestCase, NumericAssertions):
    def test_remove_extreme_values(self):

        # create some random data
        np.random.seed(10)
        test_data = np.random.random(size=(100))
        # with one clear outlier
        test_data[25] *= 15
        test_df = pd.DataFrame()
        test_df["values"] = test_data
        # now detect extremes
        cols_to_check = ["values"]
        REV = RemoveExtremeValues(cols=cols_to_check, rollingwindow=10, madthresh=10)
        data_corrected = REV.fit_transform(test_df)

        self.assertAllNaN(data_corrected.iloc[25])
        self.assertAllNotNaN(data_corrected.drop([25], axis=0))


if __name__ == "__main__":
    unittest.main()
