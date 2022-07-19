import os
import random
from re import X
import unittest
from unittest import case

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import StandardScaler
from sam.feature_engineering.base_feature_engineering import FeatureEngineer
from sam.feature_engineering.simple_feature_engineering import SimpleFeatureEngineer
from sam.models import TimeseriesMLP
from sam.visualization import sam_quantile_plot
from scipy import stats as st
from sklearn.metrics import mean_absolute_error

# If tensorflow is not available, skip these unittests
skipkeras = False
try:
    import tensorflow as tf
except ImportError:
    skipkeras = True


@pytest.mark.skipif(skipkeras, reason="Keras backend not found")
class TestTimeseriesMLP(unittest.TestCase):
    def setUp(self):
        # We are deliberately creating an extremely easy, linear problem here
        # the target is literally 17 times one of the features
        # This is because we just want to see if the model works at all, in a short time, on very
        # little data.
        # With a high enough learning rate, it should be almost perfect after a few iterations

        self.n_rows = 100
        self.train_size = int(self.n_rows * 0.8)

        self.X = pd.DataFrame(
            {
                "TIME": pd.to_datetime(np.array(range(self.n_rows)), unit="m"),
                "x": np.linspace(0, 1, self.n_rows),
            }
        ).set_index("TIME")

        self.y = 420 + 69 * self.X["x"]
        self.X_train = self.X.copy()
        self.y_train = self.y.copy()
        self.X_test = self.X.copy()
        self.y_test = self.y.copy()

        # Now start setting the RNG so we get reproducible results
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        os.environ["PYTHONHASHSEED"] = "0"

    def assert_monotonic(self, predictions, quantiles, predict_ahead):
        if isinstance(predict_ahead, int):
            predict_ahead = [predict_ahead]
        for horizon in predict_ahead:
            for q1, q2 in zip(quantiles, quantiles):
                if q1 > q2:
                    q1_col = f"predict_lead_{horizon}_q_{q1}"
                    q2_col = f"predict_lead_{horizon}_q_{q2}"
                    assert sum(predictions[q1_col] < predictions[q2_col]) == 0

    def assert_prediction(self, model, quantiles, predict_ahead, force_monotonic_quantiles):
        if isinstance(predict_ahead, int):
            predict_ahead = [predict_ahead]
        pred = model.predict(
            self.X_test,
            self.y_test,
            force_monotonic_quantiles=force_monotonic_quantiles,
        )

        actual = model.get_actual(self.y_test)

        # Check shape
        pred = pd.DataFrame(pred)
        actual = pd.DataFrame(actual)

        assert pred.shape == (self.X_test.shape[0], (len(quantiles) + 1) * len(predict_ahead))
        assert actual.shape == (self.X_test.shape[0], len(predict_ahead))

        # Check that the predictions are close to the actual values
        if force_monotonic_quantiles:
            self.assert_monotonic(pred, quantiles, predict_ahead)

    def assert_get_actual(self, model, predict_ahead, use_diff_of_y):
        if isinstance(predict_ahead, int):
            predict_ahead = [predict_ahead]
        actual = model.get_actual(self.y_test)

        if len(predict_ahead) > 1:
            assert actual.shape == (self.X_test.shape[0], len(predict_ahead))
        else:
            assert actual.shape == (self.X_test.shape[0],)

        actual = pd.DataFrame(actual)
        for i in range(len(predict_ahead)):
            horizon = predict_ahead[i]
            expected = self.y_test.shift(-horizon)
            assert_array_equal(
                actual.iloc[:, i],
                expected,
            )

    def assert_performance(
        self,
        pred,
        actual,
        predict_ahead,
        average_type,
        max_mae,
    ):
        if isinstance(predict_ahead, int):
            predict_ahead = [predict_ahead]

        pred = pd.DataFrame(pred)
        actual = pd.DataFrame(actual)

        for i in range(len(predict_ahead)):
            horizon = predict_ahead[i]
            pred_mean = pred[f"predict_lead_{horizon}_mean"]
            y_true = actual.iloc[:, i]
            missing = y_true.isna() | pred_mean.isna()
            assert mean_absolute_error(pred_mean[~missing], y_true[~missing]) <= max_mae

        if len(predict_ahead) > 1:
            for i, j in zip(range(len(predict_ahead)), range(len(predict_ahead))):
                h1 = predict_ahead[i]
                h2 = predict_ahead[j]
                if h1 > h2:
                    mae1 = mean_absolute_error(
                        pred[f"predict_lead_{h1}_{average_type}"], actual.iloc[:, i]
                    )
                    mae2 = mean_absolute_error(
                        pred[f"predict_lead_{h2}_{average_type}"], actual.iloc[:, j]
                    )
                    assert mae1 >= mae2

    @pytest.mark.parametrize(
        "predict_ahead,quantiles,average_type,use_diff_of_y,y_scaler,max_mae",
        [(0, [], False, None, 3)],
    )
    def test_model(
        self,
        predict_ahead,
        quantiles,
        average_type,
        use_diff_of_y,
        y_scaler,
        max_mae,
    ):
        fe = SimpleFeatureEngineer(keep_original=True)
        model = TimeseriesMLP(
            predict_ahead=predict_ahead,
            quantiles=quantiles,
            use_diff_of_y=use_diff_of_y,
            y_scaler=y_scaler,
            feature_engineer=fe,
            average_type=average_type,
            lr=0.01,
            epochs=40,
            verbose=0,
        )
        model.fit(self.X_train, self.y_train)
        self.assert_get_actual(model, predict_ahead, use_diff_of_y)
        self.assert_prediction(
            model=model,
            quantiles=quantiles,
            predict_ahead=predict_ahead,
            force_monotonic_quantiles=False,
        )
        self.assert_prediction(
            model=model,
            quantiles=quantiles,
            predict_ahead=predict_ahead,
            force_monotonic_quantiles=True,
        )
        self.assert_performance(
            pred=model.predict(self.X_test, self.y_test),
            predict_ahead=predict_ahead,
            actual=model.get_actual(self.y_test),
            average_type=average_type,
            max_mae=max_mae,
        )


if __name__ == "__main__":
    test = TestTimeseriesMLP()
    test.setUp()

    # "predict_ahead,quantiles,average_type,use_diff_of_y,y_scaler,max_mae"
    cases = [
        # (0, [], "mean", True, None, 3),  # should raise an error
        ([0], [0.1, 0.9], "mean", False, None, 3.0),
        (0, [], "median", False, None, 3.0),  # median prediction
        (0, [], "mean", False, StandardScaler(), 3.0),  # scaler
        (1, [], "mean", False, None, 3.0),  # forecast
        (1, [], "mean", True, None, 3.0),  # use_diff_of_y
        ([1, 2, 3], [], "mean", False, None, 3.0),  # multiforecast
        ([1, 2, 3], [], "mean", True, None, 3.0),  # multiforecast with use_diff_of_y
        (0, [0.1, 0.5, 0.9], "mean", False, None, 3.0),  # quantiles
        ([1, 2, 3], [0.1, 0.5, 0.9], "mean", False, None, 3.0),  # quantiles multiforecast
        ([1], [], "mean", True, StandardScaler(), 3.0),  # forecast, scale, use diff
        ([1, 2, 3], [0.1, 0.5, 0.9], "mean", True, StandardScaler(), 3.0),  # all options
    ]

    for c in cases:
        print("Case:", c)
        test.test_model(*c)

# TODO: FIX SCALING + USE DIFF
