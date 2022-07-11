import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sam.exploration import top_score_correlations


class TestTopCorrelation(unittest.TestCase):
    def test_equal_dataframe_output(self):
        testserie = pd.DataFrame(
            {
                "A": [1, 2, 4, 4, 3],
                "A_lag_1": [np.NaN, 1, 2, 4, 4],
                "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
                "B": [3, 3, 3, 4, 3],
                "B_lag_1": [np.NaN, 3, 3, 3, 4],
                "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
                "C": [2, 3, 1, 2, 3],
                "C_lag_1": [np.NaN, 2, 3, 1, 2],
                "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
            }
        )

        correlation_df = pd.DataFrame(
            {
                "index": ["A_lag_2", "C_lag_2"],
                "A": [-0.944911, 0.866025],
            },
            columns=["index", "A"],
        )

        assert_frame_equal(top_score_correlations(testserie, "A", score=0.8), correlation_df)

    def test_no_output(self):
        testserie = pd.DataFrame(
            {
                "A": [1, 2, 4, 4, 3],
                "A_lag_1": [np.NaN, 1, 2, 4, 4],
                "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
                "B": [3, 3, 3, 4, 3],
                "B_lag_1": [np.NaN, 3, 3, 3, 4],
                "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
                "C": [2, 3, 1, 2, 3],
                "C_lag_1": [np.NaN, 2, 3, 1, 2],
                "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
            }
        )

        correlation_df = pd.DataFrame(columns=["index", "A"])
        correlation_df["A"] = correlation_df["A"].astype(float)

        assert_frame_equal(top_score_correlations(testserie, "A", score=0.99), correlation_df)

    def test_incorrect_input(self):
        testserie = pd.DataFrame(
            {
                "A": [1, 2, 4, 4, 3],
                "A_lag_1": [np.NaN, 1, 2, 4, 4],
                "A_lag_2": [np.NaN, np.NaN, 1, 2, 4],
                "B": [3, 3, 3, 4, 3],
                "B_lag_1": [np.NaN, 3, 3, 3, 4],
                "B_lag_2": [np.NaN, np.NaN, 3, 3, 3],
                "C": [2, 3, 1, 2, 3],
                "C_lag_1": [np.NaN, 2, 3, 1, 2],
                "C_lag_2": [np.NaN, np.NaN, 2, 3, 1],
            }
        )

        self.assertRaises(Exception, top_score_correlations, testserie, "NONSENSE")


if __name__ == "__main__":
    unittest.main()
