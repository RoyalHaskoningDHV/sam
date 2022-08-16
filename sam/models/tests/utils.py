import os
import random

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.metrics import mean_absolute_error


def set_seed(func):
    """Decorator to set seeds of multiple libraries for a test"""

    def wrapper(*args, **kwargs):
        try:
            import tensorflow as tf

            tf.random.set_seed(42)
        except ImportError:
            pass

        random.seed(42)
        np.random.seed(42)
        os.environ["PYTHONHASHSEED"] = "0"
        return func(*args, **kwargs)

    return wrapper


def get_dataset():
    # We are deliberately creating an extremely easy, linear problem here
    # the target is literally 17 times one of the features
    # This is because we just want to see if the model works at all, in a short time, on very
    # little data.
    # With a high enough learning rate, it should be almost perfect after a few iterations

    n_rows = 100

    X = pd.DataFrame(
        {
            "TIME": pd.to_datetime(np.array(range(n_rows)), unit="m"),
            "x": np.linspace(0, 1, n_rows),
        }
    ).set_index("TIME")
    y = 420 + 69 * X["x"]

    return X, y


def assert_monotonic(predictions, quantiles, predict_ahead):
    if isinstance(predict_ahead, int):
        predict_ahead = [predict_ahead]
    for horizon in predict_ahead:
        for q1, q2 in zip(quantiles, quantiles):
            if q1 > q2:
                q1_col = f"predict_lead_{horizon}_q_{q1}"
                q2_col = f"predict_lead_{horizon}_q_{q2}"
                assert sum(predictions[q1_col] < predictions[q2_col]) == 0


def assert_prediction(model, X, y, quantiles, predict_ahead, force_monotonic_quantiles):
    if isinstance(predict_ahead, int):
        predict_ahead = [predict_ahead]
    pred = model.predict(X, y, force_monotonic_quantiles=force_monotonic_quantiles)

    actual = model.get_actual(y)

    # Check shape
    pred = pd.DataFrame(pred)
    actual = pd.DataFrame(actual)

    assert pred.shape == (X.shape[0], (len(quantiles) + 1) * len(predict_ahead))
    assert actual.shape == (X.shape[0], len(predict_ahead))

    # Check that the predictions are close to the actual values
    if force_monotonic_quantiles:
        assert_monotonic(pred, quantiles, predict_ahead)


def assert_get_actual(model, X, y, predict_ahead):
    if isinstance(predict_ahead, int):
        predict_ahead = [predict_ahead]
    actual = model.get_actual(y)

    if len(predict_ahead) > 1:
        assert actual.shape == (X.shape[0], len(predict_ahead))
    else:
        assert actual.shape == (X.shape[0],)

    actual = pd.DataFrame(actual)
    for i in range(len(predict_ahead)):
        horizon = predict_ahead[i]
        expected = y.shift(-horizon)
        assert_array_equal(
            actual.iloc[:, i],
            expected,
        )


def assert_performance(
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
