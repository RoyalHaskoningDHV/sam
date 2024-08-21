import unittest

import numpy as np
import pytest
from sam.metrics import (
    keras_joint_mae_tilted_loss,
    keras_joint_mse_tilted_loss,
    keras_rmse,
    keras_tilted_loss,
)

# Try importing either backend. If both fail, skip unit tests
# Unit tests will work regardless of what backend you have
skipkeras = False
try:
    import tensorflow as tf  # noqa: F401
    import tensorflow.keras.backend as K # noqa: F401
    from tensorflow.keras.layers import Input
except ImportError:
    skipkeras = True


@pytest.mark.skipif(skipkeras, reason="Keras backend not found")
class TestKerasMetrics(unittest.TestCase):
    def setUp(self):
        self.x = Input(shape=(None,))
        self.y = Input(shape=(None,))
        self.y_true = np.array([1, 0, 2, 2, 1, 0]).astype(np.float32)
        self.y_predict = np.array([2, 0, 1, 1, 1, 2]).astype(np.float32)
        # Difference: [-1, 0, 1, 1, 0, -2]

    def test_tilted_loss(self):
        result = keras_tilted_loss(self.y_true, self.y_predict, 0.8)
        # Total postitive error is 2, total negative error is 3
        expected = ((0.8 * 2) + (0.2 * 3)) / 6
        self.assertAlmostEqual(result, expected)

    def test_tilted_loss_extreme_quantile(self):
        # Quantile 0 means we don't punish true > predict at all
        result = keras_tilted_loss(self.y_true, self.y_predict, 0)
        # Total postitive error is 2, total negative error is 3
        expected = ((0 * 2) + (1 * 3)) / 6
        self.assertAlmostEqual(result.numpy(), expected)

    def test_keras_rmse(self):
        result = keras_rmse(self.y_true, self.y_predict)
        expected = np.sqrt(((self.y_true - self.y_predict) ** 2).mean())
        self.assertAlmostEqual(result, expected)

    def test_keras_sum_tilted_loss(self):
        # We test it with 2 inputs, 1 quantiles, so 4 outputs
        n_inputs = 2
        quantiles = [0.1]
        # Just 3 rows
        y_true = np.array([[1, 2], [1, 2], [1, 2]])
        y_pred = np.array([[0.5, 1.1, 1.1, 2.2], [0.5, 1.1, 1.1, 2.2], [0.5, 1.1, 1.1, 2.2]])
        # These 4 values are `quantile_1_target_1`, `quantile_1_target_2`,
        # `mean_target_1`, `mean_target_2`
        # The mean error is 0.1 for the first target, 0.2 for the second target.
        # So squared, the mse are 0.01 and 0.04, the sum is 0.05
        # The mae is 0.5 for the first, 0.9 for the second
        # But it is quantile, so multiplied by 0.1
        # So it becomes 0.05 and 0.09. The sum is 0.14
        # The total sum is 0.14 + 0.5 = 0.19
        expected = 0.19
        actual = keras_joint_mse_tilted_loss(y_true, y_pred, quantiles, n_inputs)
        self.assertAlmostEqual(expected, actual.numpy())

    def test_keras_sum_tilted_loss_single_input(self):
        # We test it with 1 inputs, 2 quantiles, so 3 outputs
        quantiles = [0.1, 0.9]

        # Just 3 rows
        y_true = np.array([[1], [1], [1]])
        y_pred = np.array([[0.5, 0.9, 0.9], [0.5, 0.9, 0.9], [0.5, 0.9, 0.9]])
        # The mean error is 0.1, so squared, 0.01
        # The mae is 0.5 for the first quantile, 0.1 for the second
        # But the first is correct, so gets multiplied by 0.1, so its 0.05
        # The second is incorrect so gets multiplied by 0.9, so its 0.09
        # The sum is 0.01 + 0.05 + 0.09 is 0.15
        expected = 0.15
        actual = keras_joint_mse_tilted_loss(y_true, y_pred, quantiles)
        self.assertAlmostEqual(expected, actual)

    def test_keras_sum_tilted_loss_2(self):
        # We test it with 2 inputs, 1 quantiles, so 4 outputs
        n_inputs = 2
        quantiles = [0.1]

        # Just 3 rows
        y_true = np.array([[1, 2], [1, 2], [1, 2]])
        y_pred = np.array([[0.5, 1.1, 1.1, 2.2], [0.5, 1.1, 1.1, 2.2], [0.5, 1.1, 1.1, 2.2]])
        # These 4 values are `quantile_1_target_1`, `quantile_1_target_2`,
        # `mean_target_1`, `mean_target_2`
        # The mean error is 0.1 for the first target, 0.2 for the second target.
        # So absolute, the mae are 0.1 and 0.2, the sum is 0.3
        # The mae is 0.5 for the first, 0.9 for the second
        # But it is quantile, so multiplied by 0.1
        # So it becomes 0.05 and 0.09. The sum is 0.14
        # The total sum is 0.3 + 0.5 = 0.44
        expected = 0.44
        actual = keras_joint_mae_tilted_loss(y_true, y_pred, quantiles, n_inputs)
        self.assertAlmostEqual(expected, actual.numpy(), places=6)

    def test_keras_sum_tilted_loss_single_input_2(self):
        # We test it with 1 inputs, 2 quantiles, so 3 outputs
        n_inputs = 1
        quantiles = [0.1, 0.9]

        # Just 3 rows
        y_true = np.array([[1], [1], [1]])
        y_pred = np.array([[0.5, 0.9, 0.9], [0.5, 0.9, 0.9], [0.5, 0.9, 0.9]])
        # The mean error is 0.1, so absolute also 0.1
        # The mae is 0.5 for the first quantile, 0.1 for the second
        # But the first is correct, so gets multiplied by 0.1, so its 0.05
        # The second is incorrect so gets multiplied by 0.9, so its 0.09
        # The sum is 0.1 + 0.05 + 0.09 is 0.24
        expected = 0.24
        actual = keras_joint_mae_tilted_loss(y_true, y_pred, quantiles, n_inputs)
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
