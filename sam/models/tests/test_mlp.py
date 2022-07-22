import os
import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sam.feature_engineering.simple_feature_engineering import SimpleFeatureEngineer
from sam.models import TimeseriesMLP
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# If tensorflow is not available, skip these unittests
skipkeras = False
try:
    import tensorflow as tf
except ImportError:
    skipkeras = True


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


@pytest.mark.skipif(skipkeras, reason="Keras backend not found")
@pytest.mark.parametrize(
    "predict_ahead,quantiles,average_type,use_diff_of_y,y_scaler,max_mae",
    [
        pytest.param((0,), (), "mean", True, None, 3, marks=pytest.mark.xfail),  # should fail
        pytest.param((0, 1), (), "mean", True, None, 3, marks=pytest.mark.xfail),  # should fail
        ((0,), (), "mean", False, None, 3),  # plain regression
        ((0,), (0.1, 0.9), "mean", False, None, 3.0),  # quantile regression
        ((0,), (), "median", False, None, 3.0),  # median prediction
        ((0,), (), "mean", False, StandardScaler(), 3.0),  # scaler
        ((1,), (), "mean", False, None, 3.0),  # forecast
        ((1,), (), "mean", True, None, 3.0),  # use_diff_of_y
        ((1, 2, 3), (), "mean", False, None, 3.0),  # multiforecast
        ((1, 2, 3), (), "mean", True, None, 3.0),  # multiforecast with use_diff_of_y
        ((0,), (0.1, 0.5, 0.9), "mean", False, None, 3.0),  # quantiles
        ((1, 2, 3), (0.1, 0.5, 0.9), "mean", False, None, 3.0),  # quantiles multiforecast
        ((1,), (), "mean", True, StandardScaler(), 3.0),  # all options except quantiles
        ((1, 2, 3), (0.1, 0.5, 0.9), "mean", True, StandardScaler(), 3.0),  # all options
    ],
)
def test_mlp(
    predict_ahead,
    quantiles,
    average_type,
    use_diff_of_y,
    y_scaler,
    max_mae,
):

    # Now start setting the RNG so we get reproducible results
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ["PYTHONHASHSEED"] = "0"

    X, y = get_dataset()

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
    model.fit(X, y)
    assert_get_actual(model, X, y, predict_ahead)
    assert_prediction(
        model=model,
        X=X,
        y=y,
        quantiles=quantiles,
        predict_ahead=predict_ahead,
        force_monotonic_quantiles=False,
    )
    assert_prediction(
        model=model,
        X=X,
        y=y,
        quantiles=quantiles,
        predict_ahead=predict_ahead,
        force_monotonic_quantiles=True,
    )
    assert_performance(
        pred=model.predict(X, y),
        predict_ahead=predict_ahead,
        actual=model.get_actual(y),
        average_type=average_type,
        max_mae=max_mae,
    )
