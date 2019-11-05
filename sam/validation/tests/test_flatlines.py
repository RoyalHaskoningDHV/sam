import unittest
from sam.validation import RemoveFlatlines
from .numeric_assertions import NumericAssertions
import pandas as pd


class TestRemoveExtremes(unittest.TestCase, NumericAssertions):

    def test_remove_flatlines(self):

        # create some random data
        data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
        test_df = pd.DataFrame()
        test_df['values'] = data
        # now detect flatlines
        cols_to_check = ['values']
        RF = RemoveFlatlines(
            cols=cols_to_check,
            window=2)
        data_corrected = RF.fit_transform(test_df)

        self.assertAllNaN(data_corrected.iloc[[4, 5, 6]])
        self.assertAllNotNaN(data_corrected.drop([4, 5, 6], axis=0))


if __name__ == '__main__':
    unittest.main()
