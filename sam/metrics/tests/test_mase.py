import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sam.metrics import mean_absolute_scaled_error


class TestMASE(unittest.TestCase):
    def test_mase_1d(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        self.assertAlmostEqual(mean_absolute_scaled_error(y_true, y_pred, 1), 0.1)
        self.assertAlmostEqual(mean_absolute_scaled_error(y_true, y_pred, 2), 0.05)
        self.assertAlmostEqual(mean_absolute_scaled_error(y_true, y_pred, 4), 0.025)

    def test_mase_2d(self):
        y_true = [[1, 10], [2, 20], [3, 30], [4, 40]]
        y_pred = [[1.1, 11], [2.1, 21], [3.1, 31], [4.1, 41]]
        sample_weights = [0.5, 0.3, 0.1, 0.1]

        output = mean_absolute_scaled_error(y_true, y_pred, 1, sample_weights)
        assert_almost_equal(output, np.array([0.1]))

        output = mean_absolute_scaled_error(y_true, y_pred, 1, sample_weights, "raw_values")
        assert_almost_equal(output, np.array([0.1, 0.1]))

    def test_mase_incorrect(self):

        self.assertRaises(ValueError, mean_absolute_scaled_error, [1], [2], 1)
        self.assertRaises(ValueError, mean_absolute_scaled_error, [1, 2, 3], [1, 2, 3], 0)
        self.assertRaises(ValueError, mean_absolute_scaled_error, [1, 2, 3], [1, 2, 3], 1.5)
        self.assertRaises(ValueError, mean_absolute_scaled_error, [1, 1, 1], [1, 2, 3], 1)
        self.assertRaises(
            ValueError,
            mean_absolute_scaled_error,
            [[1, 2], [1, 3]],
            [[1, 2], [1, 3]],
            1,
            multioutput="raw_values",
        )


if __name__ == "__main__":
    unittest.main()
