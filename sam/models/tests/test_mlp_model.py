import unittest
from pathlib import Path
import numpy as np
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

PATH = __file__


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


class TestOptimizer(unittest.TestCase):
    def test_default_optimizer(self):
        from keras.src.optimizers import Adam

        X, y = get_dataset()
        fe = SimpleFeatureEngineer(keep_original=True)
        model = MLPTimeseriesRegressor(epochs=1, feature_engineer=fe)
        model.fit(X, y)
        self.assertIsInstance(model.model_.optimizer, Adam)
        self.assertAlmostEqual(
            float(model.model_.optimizer.learning_rate.value.value()),
            0.001,
        )

    def test_set_learning_rate(self):
        from keras.src.optimizers import Adam

        X, y = get_dataset()
        fe = SimpleFeatureEngineer(keep_original=True)
        learning_rate = 0.123
        model = MLPTimeseriesRegressor(epochs=1, lr=learning_rate, feature_engineer=fe)
        model.fit(X, y)
        self.assertIsInstance(model.model_.optimizer, Adam)
        self.assertAlmostEqual(
            float(model.model_.optimizer.learning_rate.value.value()),
            learning_rate,
        )

    def test_overwrite_optimizer(self):
        from keras.src.optimizers import AdamW

        X, y = get_dataset()
        fe = SimpleFeatureEngineer(keep_original=True)
        learning_rate = 0.01
        weight_decay = 1e-5
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        model = MLPTimeseriesRegressor(epochs=1, feature_engineer=fe, optimizer=optimizer)
        model.fit(X, y)
        self.assertIsInstance(model.model_.optimizer, AdamW)
        self.assertAlmostEqual(
            float(model.model_.optimizer.learning_rate.value.value()),
            learning_rate,
        )
        self.assertAlmostEqual(
            float(model.model_.optimizer.weight_decay),
            weight_decay,
        )


class TestPipelineFeatureEngineer(unittest.TestCase):
    def test_get_feature_names_out(self):
        from sklearn.pipeline import Pipeline
        from sam.feature_engineering import BuildRollingFeatures
        X, y = get_dataset()
        fe = Pipeline([
            ("roll", BuildRollingFeatures(window_size='1h')),
            ("scaler", StandardScaler())
        ])
        model = MLPTimeseriesRegressor(epochs=1, feature_engineer=fe)
        model.fit(X, y)
        feature_names = model.get_feature_names_out()
        self.assertListEqual(list(feature_names), ['x', 'x#mean_1h'])


class TestLoadDump(unittest.TestCase):
    file_dir = Path(PATH).parent / "files"

    @classmethod
    def setUpClass(cls):
        import os
        os.makedirs(cls.file_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import os

        files = os.listdir(cls.file_dir)
        for file in files:
            os.remove(cls.file_dir / file)
        os.removedirs(cls.file_dir)

    def test_dump_load_parameters(self):
        import onnxruntime as ort
        import keras

        X, y = get_dataset()
        fe = SimpleFeatureEngineer(keep_original=True)
        model = MLPTimeseriesRegressor(epochs=1, feature_engineer=fe)
        model.fit(X, y)

        model.dump_parameters(foldername=self.file_dir, file_extension='.onnx')
        y_pred_tf = model.predict(X=X)
        self.assertIsInstance(model.model_, keras.Model)
        model.model_ = model.load_parameters(obj=model, foldername=self.file_dir)

        self.assertIsInstance(model.model_, ort.InferenceSession)
        y_pred_onnx = model.predict(X=X)

        self.assertTrue(np.all(np.isclose(y_pred_onnx.values, y_pred_tf.values)))

    def test_to_from_dict(self):
        import onnxruntime as ort
        import keras

        X, y = get_dataset()
        fe = SimpleFeatureEngineer(keep_original=True)
        model = MLPTimeseriesRegressor(epochs=1, feature_engineer=fe, y_scaler=StandardScaler())
        model.fit(X, y)

        model.dump_parameters(foldername=self.file_dir, file_extension='.onnx')
        params = model.to_dict()
        y_pred_tf = model.predict(X=X)
        self.assertIsInstance(model.model_, keras.Model)

        model = MLPTimeseriesRegressor.from_dict(params=params)
        model.model_ = model.load_parameters(obj=model, foldername=self.file_dir)
        self.assertIsInstance(model.model_, ort.InferenceSession)
        y_pred_onnx = model.predict(X=X)

        self.assertTrue(np.all(np.isclose(y_pred_onnx.values, y_pred_tf.values)))
