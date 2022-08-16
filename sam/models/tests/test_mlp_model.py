import pytest
from sam.feature_engineering.simple_feature_engineering import SimpleFeatureEngineer
from sam.models import MLPTimeseriesRegressor
from sam.models.tests.utils import (
    assert_get_actual,
    assert_performance,
    assert_prediction,
    get_dataset,
    set_seed,
)
from sklearn.preprocessing import StandardScaler

# If tensorflow is not available, skip these unittests
skipkeras = False
try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    skipkeras = True


@set_seed
def train_mlp(X, y, predict_ahead, quantiles, average_type, use_diff_of_y, y_scaler):
    fe = SimpleFeatureEngineer(keep_original=True)
    model = MLPTimeseriesRegressor(
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
    return model


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

    X, y = get_dataset()
    model = train_mlp(X, y, predict_ahead, quantiles, average_type, use_diff_of_y, y_scaler)
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
