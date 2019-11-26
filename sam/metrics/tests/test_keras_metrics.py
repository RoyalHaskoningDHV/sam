from sam.metrics import keras_tilted_loss, keras_rmse, keras_joint_mse_tilted_loss
import numpy as np
import unittest
import pytest

# Try importing either backend. If both fail, skip unit tests
# Unit tests will work regardless of what backend you have
skipkeras = False
try:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input
    import tensorflow as tf
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
        expected = np.sqrt(((self.y_true-self.y_predict) ** 2).mean())
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
