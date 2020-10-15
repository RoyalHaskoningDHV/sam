import unittest
from sam.validation import RemoveFlatlines
from sam.validation.tests import NumericAssertions
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

    def test_remove_flatlines_auto_low(self):

        # create some random data
        data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
        test_df = pd.DataFrame()
        test_df['values'] = data
        # now detect flatlines with high tolerance (low pvalues)
        cols_to_check = ['values']
        RF = RemoveFlatlines(
            cols=cols_to_check,
            window='auto',
            pvalue=1e-100)
        data_corrected = RF.fit_transform(test_df)

        # no flatlines should be detected
        self.assertAllNotNaN(data_corrected)
        pd.testing.assert_frame_equal(test_df, data_corrected)

    def test_remove_flatlines_auto_high(self):

        # create some random data
        data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
        test_df = pd.DataFrame()
        test_df['values'] = data
        # now detect flatlines with low tolerance (high pvalues)
        cols_to_check = ['values']
        RF = RemoveFlatlines(
            cols=cols_to_check,
            window='auto',
            pvalue=0.999)
        data_corrected = RF.fit_transform(test_df)

        # all flatlines should be removed
        self.assertAllNaN(data_corrected.iloc[[4, 5, 6, 9, 10]])
        self.assertAllNotNaN(data_corrected.drop([4, 5, 6, 9, 10], axis=0))
        pd.testing.assert_frame_equal(
            test_df.drop([4, 5, 6, 9, 10], axis=0),
            data_corrected.drop([4, 5, 6, 9, 10], axis=0)
        )


if __name__ == '__main__':
    unittest.main()
