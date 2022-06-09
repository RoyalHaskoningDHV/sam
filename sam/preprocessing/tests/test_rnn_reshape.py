import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sam.preprocessing import RecurrentReshaper


class TestRecurrentReshaper(unittest.TestCase):
    def setUp(self):
        self.window = 2
        self.lookback = 1
        self.X_in = pd.DataFrame({"x1": [1.1, 2.1, 1.2, 1.1], "x2": [7.7, 6.7, 7.6, 6.6]})
        self.X_out = np.array(
            [
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [1.1, 7.7]],
                [[1.1, 7.7], [2.1, 6.7]],
                [[2.1, 6.7], [1.2, 7.6]],
            ]
        )
        self.X_empty = np.empty((0, self.window, 0))

    def test_reshape_features_rnn(self):
        reshaper = RecurrentReshaper(window=self.window, lookback=self.lookback)
        result = reshaper.fit_transform(self.X_in)
        assert_array_equal(result, self.X_out)

    def test_reshape_features_rnn_empty(self):
        reshaper = RecurrentReshaper(window=self.window, lookback=self.lookback)
        result = reshaper.fit_transform(pd.DataFrame())
        assert_array_equal(result, self.X_empty)


if __name__ == "__main__":
    unittest.main()
