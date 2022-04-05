import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sam.utils import make_df_monotonic, sum_grouped_columns


class TestSumGroupedColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "X#_2": [1, 2, 3],
                "X#_5": [3, np.nan, 5],
                "y": [9, np.nan, 7],
                "X": [10, 11, 12],
            },
            columns=["X#_2", "X#_5", "y", "X"],
            index=[4, 6, 7],
        )
        self.dfbackup = self.df.copy()

    def test_two_groups(self):
        result = sum_grouped_columns(self.df)
        expected = pd.DataFrame(
            {"X": [14, 13, 20], "y": [9, 0, 7]},
            columns=["X", "y"],
            index=[4, 6, 7],
            dtype=float,
        )

        assert_frame_equal(result, expected)
        assert_frame_equal(self.df, self.dfbackup)

    def test_other_options(self):
        result = sum_grouped_columns(self.df, sep="_", skipna=False)
        expected = pd.DataFrame(
            {"X": [10, 11, 12], "X#": [4, np.nan, 8], "y": [9, np.nan, 7]},
            columns=["X", "X#", "y"],
            index=[4, 6, 7],
            dtype=float,
        )

        assert_frame_equal(result, expected)
        assert_frame_equal(self.df, self.dfbackup)


class TestMakeDataFrameMonotonic(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "predict_lead_0_q_2.866515719235352e-07": [-5, -5, -5],
                "predict_lead_0_q_3.167124183311998e-05": [-3, -3, -3],
                "predict_lead_0_q_0.0002326290790355401": [0, 0, 0],
                "predict_lead_0_q_0.0013498980316301035": [2, 2, 2],
                "predict_lead_0_q_0.02275013194817921": [3, 3, 3],
                "predict_lead_0_q_0.15865525393145707": [5, 5, 5],
            },
            dtype=float,
        )

    def test_increasing(self):
        result = make_df_monotonic(self.df)

        assert_frame_equal(result, self.df)

    def test_decreasing(self):
        reversed_df = self.df[self.df.columns[::-1]]
        result = make_df_monotonic(reversed_df, aggregate_func="min")

        assert_frame_equal(result, reversed_df)

    def test_unknown_order(self):
        self.assertRaises(ValueError, make_df_monotonic, self.df, aggregate_func="unknown")

    def test_increasing_overlapping_quantile(self):
        self.df.iloc[1, 0] = -2
        result = make_df_monotonic(self.df)
        expected = pd.DataFrame(
            {
                "predict_lead_0_q_2.866515719235352e-07": [-5, -2, -5],
                "predict_lead_0_q_3.167124183311998e-05": [-3, -2, -3],
                "predict_lead_0_q_0.0002326290790355401": [0, 0, 0],
                "predict_lead_0_q_0.0013498980316301035": [2, 2, 2],
                "predict_lead_0_q_0.02275013194817921": [3, 3, 3],
                "predict_lead_0_q_0.15865525393145707": [5, 5, 5],
            },
            dtype=float,
        )

        assert_frame_equal(result, expected)

    def test_descending_overlapping_quantile(self):
        self.df.iloc[1, 1] = -8
        result = make_df_monotonic(self.df, aggregate_func="min")

        expected = pd.DataFrame(
            {
                "predict_lead_0_q_2.866515719235352e-07": [-5, -5, -5],
                "predict_lead_0_q_3.167124183311998e-05": [-5, -8, -5],
                "predict_lead_0_q_0.0002326290790355401": [-5, -8, -5],
                "predict_lead_0_q_0.0013498980316301035": [-5, -8, -5],
                "predict_lead_0_q_0.02275013194817921": [-5, -8, -5],
                "predict_lead_0_q_0.15865525393145707": [-5, -8, -5],
            },
            dtype=float,
        )

        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
