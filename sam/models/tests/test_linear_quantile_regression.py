import pandas as pd
import numpy as np
import random
import os
from sam.models import LinearQuantileRegression
import unittest
from numpy.testing import assert_almost_equal


class TestLinearQuantileRegression(unittest.TestCase):

    def setUp(self):
        # We are deliberately creating an extremely easy, linear problem here
        # the target is literally 17 times one of the features
        # This is because we just want to see if the model works at all, in a short time, on very
        # little data.
        # With a high enough learning rate, it should be almost perfect after a few iterations

        self.n_rows = 100
        self.train_size = int(self.n_rows * 0.8)

        self.X = pd.DataFrame({
            'x1': np.arange(1000, 2000)
        })

        # Now start setting the RNG so we get reproducible results
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '0'

        self.y = 42 + self.X['x1'] * 13 + np.random.uniform(low=-10, high=10, size=1000)

    def test_simpel_linear_quantile_regression(self):

        model = LinearQuantileRegression(quantile=0.5)

        model.fit(self.X, self.y)

        assert_almost_equal(model.model_result_.params.x1 / 13, 1, decimal=1)
        assert_almost_equal(model.model_result_.params.const / 42, 1, decimal=1)


if __name__ == '__main__':
    unittest.main()
