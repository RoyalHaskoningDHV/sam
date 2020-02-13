import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
from sam.metrics import train_mean_r2


class TestTrainMeanR2(unittest.TestCase):

    def test_train_mean_r2(self):

        from sklearn.metrics import r2_score
        from sam.metrics import train_mean_r2

        np.random.seed(42)
        N = 1000
        model = np.zeros(N)
        for f in [0.001, 0.005, 0.01, 0.05]:
            for p in [0, np.pi/4]:
                model += (1/f) * np.sin(2*np.pi*f*np.arange(N) + p)
        data = model + np.random.normal(scale=250, size=N)

        custom_r2s = []
        keras_r2s = []
        for test_ratio in [0.8, 0.5, 0.2]:

            test_n = int(test_ratio * N)
            train_n = N-test_n
            train_data = data[:train_n]
            test_data = data[train_n:]
            train_pred = model[:train_n]
            test_pred = model[train_n:]

            keras_r2s.append(r2_score(test_data, test_pred))
            custom_r2s.append(train_mean_r2(test_data, test_pred, np.mean(train_data)))

        # keras r2 should decrease with decreasing test size, custom r2 should do so less
        assert_array_almost_equal(keras_r2s, [0.962522, 0.894608, 0.734713])
        assert_array_almost_equal(custom_r2s, [0.987602, 0.989454, 0.870157])


if __name__ == '__main__':
    unittest.main()
