import unittest
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sam.utils import sum_grouped_columns


class TestDataframeFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'X#_2': [1, 2, 3],
            'X#_5': [3, np.nan, 5],
            'y': [9, np.nan, 7],
            'X': [10, 11, 12]
        }, columns=['X#_2', 'X#_5', 'y', 'X'], index=[4, 6, 7])
        self.dfbackup = self.df.copy()

    def test_two_groups(self):
        result = sum_grouped_columns(self.df)
        expected = pd.DataFrame({
            'X': [14, 13, 20],
            'y': [9, 0, 7]
        }, columns=['X', 'y'], index=[4, 6, 7], dtype=float)

        assert_frame_equal(result, expected)
        assert_frame_equal(self.df, self.dfbackup)

    def test_other_options(self):
        result = sum_grouped_columns(self.df, sep='_', skipna=False)
        expected = pd.DataFrame({
            'X': [10, 11, 12],
            'X#': [4, np.nan, 8],
            'y': [9, np.nan, 7]
        }, columns=['X', 'X#', 'y'], index=[4, 6, 7], dtype=float)

        assert_frame_equal(result, expected)
        assert_frame_equal(self.df, self.dfbackup)

if __name__ == '__main__':
    unittest.main()
