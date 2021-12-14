import os
import random
import unittest

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from sam.metrics import tilted_loss
from sam.models import LinearQuantileRegression

# Is statsmodels not available, skip these unittests
skipstatsmodels = False
try:
    from statsmodels.regression.quantile_regression import QuantReg  # noqa: F401
except ImportError:
    skipstatsmodels = True


@pytest.mark.skipif(skipstatsmodels, reason="Statsmodels not found")
class TestLinearQuantileRegression(unittest.TestCase):
    def setUp(self):
        # We are deliberately creating an extremely easy regression problem here
        # The output y is a linear combination of the input + some uniform noise
        # The model should fit a 50% quantile almost perfectly
        # This is because we just want to see if the model works at all.

        self.n_rows = 100
        self.train_size = int(self.n_rows * 0.8)

        self.X = pd.DataFrame({"x1": np.arange(1000, 2000)})

        # Now start setting the RNG so we get reproducible results
        random.seed(42)
        np.random.seed(42)
        os.environ["PYTHONHASHSEED"] = "0"

        self.y = 42 + self.X["x1"] * 13 + np.random.uniform(low=-10, high=10, size=1000)

    def test_simpel_linear_quantile_regression(self):

        model = LinearQuantileRegression(quantiles=[0.5])

        model.fit(self.X, self.y)
        pred = model.predict(self.X)
        score = model.score(self.X, self.y)

        assert_almost_equal(model.coef_[0].x1 / 13, 1, decimal=1)
        assert_almost_equal(model.coef_[0].const / 42, 1, decimal=1)

        assert_almost_equal(model.coef_[0].x1 / 13, 1, decimal=1)
        assert_almost_equal(model.coef_[0].const / 42, 1, decimal=1)

        self.assertEqual(model.prediction_cols, ["predict_q_0.5"])
        self.assertEqual(pred.columns, ["predict_q_0.5"])

        self.assertLess(model.score(self.X, self.y), 20)
        assert_almost_equal(score, tilted_loss(self.y, pred["predict_q_0.5"]))


if __name__ == "__main__":
    unittest.main()
