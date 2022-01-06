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


def _train_quantile_regression(X, y, quantiles=[0.5]):
    """
    Train a linear quantile regression model. For testing purposes
    """
    model = LinearQuantileRegression(quantiles=quantiles)
    model.fit(X, y)
    pred = model.predict(X)
    score = model.score(X, y)
    return model, pred, score


@pytest.mark.skipif(skipstatsmodels, reason="Statsmodels not found")
class TestLinearQuantileRegression(unittest.TestCase):
    def setUp(self):
        # We are deliberately creating an extremely easy regression problem here
        # The output y is a linear combination of the input + some uniform noise
        # The model should fit a 50% quantile almost perfectly
        # This is because we just want to see if the model works at all.

        # First setting the RNG so we get reproducible results
        random.seed(42)
        np.random.seed(42)
        os.environ["PYTHONHASHSEED"] = "0"

        # Create some example data
        self.X = pd.DataFrame({"x1": np.arange(1000, 2000)})
        err = np.random.uniform(low=-10, high=10, size=1000)  # random noise
        self.y = 42 + self.X["x1"] * 13 + err

    def test_linear_quantile_regression(self, use_numpy=False):
        if use_numpy:
            X, y = self.X.values, self.y.values
            colname = "X1"
        else:
            X, y = self.X, self.y
            colname = "x1"

        model, pred, score = _train_quantile_regression(X, y)

        # test if fitted coefficients are as expected for example
        assert_almost_equal(model.coef_[0][colname] / 13, 1, decimal=1)
        assert_almost_equal(model.coef_[0]["const"] / 42, 1, decimal=1)
        assert_almost_equal(model.coef_[0][colname] / 13, 1, decimal=1)
        assert_almost_equal(model.coef_[0]["const"] / 42, 1, decimal=1)

        # expected performance
        self.assertLess(model.score(X, y), 20)
        assert_almost_equal(score, tilted_loss(y, pred["predict_q_0.5"]))

        # expected output format
        self.assertEqual(model.prediction_cols, ["predict_q_0.5"])
        self.assertEqual(pred.columns, ["predict_q_0.5"])

    def test_linear_quantile_regression_numpy(self):
        self.test_linear_quantile_regression(use_numpy=True)


if __name__ == "__main__":
    unittest.main()
