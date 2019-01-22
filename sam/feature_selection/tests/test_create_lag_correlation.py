import unittest
from pandas.testing import assert_frame_equal

import numpy as np
import pandas as pd
from sam.feature_selection.lag_correlation import create_lag_correlation


class TestCreateLagCorrelation(unittest.TestCase):

    def test_dataframe_output(self):
        testserie = pd.DataFrame({'TEST_lag_0': [0, 1, 2, 3],
                                  'TEST_lag_1': [np.NaN, 0, 1, 2],
                                  'TEST_lag_2': [np.NaN, np.NaN, 0, 1],
                                  'OTHER_lag_0': [3, 2, 1, 0],
                                  'OTHER_lag_1': [np.NaN, 3, 2, 1],
                                  'OTHER_lag_2': [np.NaN, np.NaN, 3, 2]
                                  })
        output_df = pd.DataFrame({
            'LAG': [0, 1, 2],
            'OTHER': [-1.0, -1.0, -1.0],
            'TEST': [1.0, 1.0, 1.0],
        })

        assert_frame_equal(create_lag_correlation(testserie, 'TEST_lag_0'),
                           output_df)

        self.assertRaises(Exception,
                          create_lag_correlation,
                          testserie,
                          'NONSENSE')


if __name__ == '__main__':
    unittest.main()
