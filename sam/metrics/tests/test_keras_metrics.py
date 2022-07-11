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

    # Necessary for shap DeepExplainer, see: https://github.com/slundberg/shap/issues/2189
    tf.compat.v1.disable_v2_behavior()
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input
except ImportError:
    skipkeras = True


# Helper function to create keras function that will actually compute
# the result. Since tensorflow uses lazy evaluation by default.
# x and y are (empty) tensors, fun is a function that accepts 2 tensors as input
def create_function(fun, x, y, param=None):
    # Create a keras function
    if param is not None:
        func = K.function([x, y], fun(x, y, param))
    else:
        func = K.function([x, y], fun(x, y))

    # create an easy-to-use wrapper for this function
    def kerasfun(y_true, y_pred):
        # func expects a list of two 2D arrays
        # we have 2 1D arrays instead, so we reshape it
        foo = [np.reshape(y_true, (1, -1)), np.reshape(y_pred, (1, -1))]
        # func gives an array as output. We just want a scalar
        return func(foo)[0]

    return kerasfun


@pytest.mark.skipif(skipkeras, reason="Keras backend not found")
class TestKerasMetrics(unittest.TestCase):
    def setUp(self):
        self.x = Input(shape=(None,))
        self.y = Input(shape=(None,))
        self.y_true = np.array([1, 0, 2, 2, 1, 0])
        self.y_predict = np.array([2, 0, 1, 1, 1, 2])
        # Difference: [-1, 0, 1, 1, 0, -2]

    def test_tilted_loss(self):
        fun = create_function(keras_tilted_loss, self.x, self.y, 0.8)
        result = fun(self.y_true, self.y_predict)
        # Total postitive error is 2, total negative error is 3
        expected = ((0.8 * 2) + (0.2 * 3)) / 6
        self.assertAlmostEqual(result, expected)

    def test_tilted_loss_extreme_quantile(self):
        # Quantile 0 means we don't punish true > predict at all
        fun = create_function(keras_tilted_loss, self.x, self.y, 0)
        result = fun(self.y_true, self.y_predict)
        # Total postitive error is 2, total negative error is 3
        expected = ((0 * 2) + (1 * 3)) / 6
        self.assertAlmostEqual(result, expected)

    def test_keras_rmse(self):
        fun = create_function(keras_rmse, self.x, self.y)
        result = fun(self.y_true, self.y_predict)
        expected = np.sqrt(((self.y_true - self.y_predict) ** 2).mean())
        self.assertAlmostEqual(result, expected)

    def test_keras_sum_tilted_loss(self):
        # We test it with 2 inputs, 1 quantiles, so 4 outputs
        n_inputs = 2
        quantiles = [0.1]
        n_outputs = (len(quantiles) + 1) * n_inputs
        x = Input(shape=((n_inputs,)))
        y = Input(shape=((n_outputs,)))
        func = K.function([x, y], keras_joint_mse_tilted_loss(x, y, quantiles, n_inputs))

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
        actual = func([y_true, y_pred])
        self.assertAlmostEqual(expected, actual)

    def test_keras_sum_tilted_loss_single_input(self):
        # We test it with 1 inputs, 2 quantiles, so 3 outputs
        n_inputs = 1
        quantiles = [0.1, 0.9]
        n_outputs = len(quantiles) + 1
        x = Input(shape=((n_inputs,)))
        y = Input(shape=((n_outputs,)))
        func = K.function([x, y], keras_joint_mse_tilted_loss(x, y, quantiles))

        # Just 3 rows
        y_true = np.array([[1], [1], [1]])
        y_pred = np.array([[0.5, 0.9, 0.9], [0.5, 0.9, 0.9], [0.5, 0.9, 0.9]])
        # The mean error is 0.1, so squared, 0.01
        # The mae is 0.5 for the first quantile, 0.1 for the second
        # But the first is correct, so gets multiplied by 0.1, so its 0.05
        # The second is incorrect so gets multiplied by 0.9, so its 0.09
        # The sum is 0.01 + 0.05 + 0.09 is 0.15
        expected = 0.15
        actual = func([y_true, y_pred])
        self.assertAlmostEqual(expected, actual)

    def test_keras_sum_tilted_loss_2(self):
        # We test it with 2 inputs, 1 quantiles, so 4 outputs
        n_inputs = 2
        quantiles = [0.1]
        n_outputs = (len(quantiles) + 1) * n_inputs
        x = Input(shape=((n_inputs,)))
        y = Input(shape=((n_outputs,)))
        func = K.function([x, y], keras_joint_mae_tilted_loss(x, y, quantiles, n_inputs))

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
        actual = func([y_true, y_pred])
        self.assertAlmostEqual(expected, actual, places=6)

    def test_keras_sum_tilted_loss_single_input_2(self):
        # We test it with 1 inputs, 2 quantiles, so 3 outputs
        n_inputs = 1
        quantiles = [0.1, 0.9]
        n_outputs = len(quantiles) + 1
        x = Input(shape=((n_inputs,)))
        y = Input(shape=((n_outputs,)))
        func = K.function([x, y], keras_joint_mae_tilted_loss(x, y, quantiles))

        # Just 3 rows
        y_true = np.array([[1], [1], [1]])
        y_pred = np.array([[0.5, 0.9, 0.9], [0.5, 0.9, 0.9], [0.5, 0.9, 0.9]])
        # The mean error is 0.1, so absolute also 0.1
        # The mae is 0.5 for the first quantile, 0.1 for the second
        # But the first is correct, so gets multiplied by 0.1, so its 0.05
        # The second is incorrect so gets multiplied by 0.9, so its 0.09
        # The sum is 0.1 + 0.05 + 0.09 is 0.24
        expected = 0.24
        actual = func([y_true, y_pred])
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
