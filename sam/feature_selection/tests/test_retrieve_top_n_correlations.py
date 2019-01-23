import unittest
from pandas.testing import assert_frame_equal

import numpy as np
import pandas as pd
from sam.feature_selection.top_correlation import retrieve_top_n_correlations


class TestRetrieveTopNCorrelation(unittest.TestCase):

    def test_dataframe_group_false_output(self):
        testserie = pd.DataFrame({'TEST_lag_0': [0, 1, 2, 3],
                                  'TEST_lag_1': [np.NaN, 0, 1, 2],
                                  'TEST_lag_2': [np.NaN, np.NaN, 0, 1],
                                  'OTHER_lag_0': [3, 2, 1, 0],
                                  'OTHER_lag_1': [np.NaN, 3, 2, 1],
                                  'OTHER_lag_2': [np.NaN, np.NaN, 3, 2]
                                  })
        lagserie = pd.DataFrame({
            'index': ['TEST_lag_1'],
            'GROUP': ['TEST'],
            'TEST_lag_0': [1.0],
        }, columns=['index', 'GROUP', 'TEST_lag_0'])

        assert_frame_equal(retrieve_top_n_correlations(testserie,
                                                       'TEST_lag_0',
                                                       grouped=False,
                                                       n=1),
                           lagserie)

    def test_dataframe_group_true_output(self):
        testserie = pd.DataFrame({'TEST_lag_0': [0, 1, 2, 3],
                                  'TEST_lag_1': [np.NaN, 0, 1, 2],
                                  'TEST_lag_2': [np.NaN, np.NaN, 0, 1],
                                  'OTHER_lag_0': [3, 2, 1, 0],
                                  'OTHER_lag_1': [np.NaN, 3, 2, 1],
                                  'OTHER_lag_2': [np.NaN, np.NaN, 3, 2]
                                  })

        lagserie = pd.DataFrame({
            'GROUP': ['OTHER', 'TEST'],
            'index': ['OTHER_lag_0', 'TEST_lag_1'],
            'TEST_lag_0': [-1.0, 1.0],
        }, columns=['GROUP', 'index', 'TEST_lag_0'])

        assert_frame_equal(retrieve_top_n_correlations(testserie,
                                                       'TEST_lag_0',
                                                       grouped=True,
                                                       n=1),
                           lagserie)

    def test_random_group_true_output(self):
        testserie = pd.DataFrame({
            "A": [1, 2, 4, 4, 3],
            "A_lag_1": [np.NaN, 1, 2, 4, 4],
            "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
            "B": [3, 3, 3, 4, 3],
            "B_lag_1": [np.NaN, 3, 3, 3, 4],
            "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
            "C": [2, 3, 1, 2, 3],
            "C_lag_1": [np.NaN, 2, 3, 1, 2],
            "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
        })

        correlation_df = pd.DataFrame({
            'GROUP': ['A', 'A', 'B', 'B', 'C', 'C'],
            'index': ['A_lag_2', 'A_lag_1', 'B', 'B_lag_1', 'C_lag_2', 'C'],
            'A': [-0.944911, 0.522233, 0.514496, -0.174078, 0.866025, -0.412514],
        }, columns=['GROUP', 'index', 'A'])

        assert_frame_equal(retrieve_top_n_correlations(testserie,
                                                       'A',
                                                       2,
                                                       grouped=True),
                           correlation_df
                           )

    def test_random_group_false_output(self):
        testserie = pd.DataFrame({
            "A": [1, 2, 4, 4, 3],
            "A_lag_1": [np.NaN, 1, 2, 4, 4],
            "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
            "B": [3, 3, 3, 4, 3],
            "B_lag_1": [np.NaN, 3, 3, 3, 4],
            "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
            "C": [2, 3, 1, 2, 3],
            "C_lag_1": [np.NaN, 2, 3, 1, 2],
            "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
        })

        correlation_df = pd.DataFrame({
            'index': ['A_lag_2', 'C_lag_2'],
            'GROUP': ['A', 'C'],
            'A': [-0.944911, 0.866025],
        }, columns=['index', 'GROUP', 'A'])

        assert_frame_equal(retrieve_top_n_correlations(testserie,
                                                       'A',
                                                       2,
                                                       grouped=False),
                           correlation_df
                           )

    def test_incorrect_input(self):
        testserie = pd.DataFrame({
            "A": [1, 2, 4, 4, 3],
            "A_lag_1": [np.NaN, 1, 2, 4, 4],
            "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
            "B": [3, 3, 3, 4, 3],
            "B_lag_1": [np.NaN, 3, 3, 3, 4],
            "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
            "C": [2, 3, 1, 2, 3],
            "C_lag_1": [np.NaN, 2, 3, 1, 2],
            "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
        })

        self.assertRaises(Exception,
                          retrieve_top_n_correlations,
                          testserie,
                          'NONSENSE')

if __name__ == '__main__':
    unittest.main()
