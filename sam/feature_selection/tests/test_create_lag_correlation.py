import unittest
from pandas.testing import assert_frame_equal

import pandas as pd
from sam.feature_selection.lag_correlation import create_lag_correlation


class TestCreateLagCorrelation(unittest.TestCase):

    def test_dataframe_output(self):
        X = pd.DataFrame({
            'TEST': [0, 1, 2, 3],
            'TARGET': [3, 2, 1, 0]
        })

        expected = pd.DataFrame({
            'LAG': [0, 1, 2],
            'TEST': [-1.0, -1.0, -1.0],
        })

        result = create_lag_correlation(X, 'TARGET', lag=3)
        assert_frame_equal(result, expected)

        self.assertRaises(Exception, create_lag_correlation, X, 'nonsense')


if __name__ == '__main__':
    unittest.main()
