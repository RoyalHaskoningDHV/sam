import numpy as np
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from sam.feature_engineering import SimpleFeatureEngineer


class TestSimpleFeatureEngineer(unittest.TestCase):
    dates = pd.date_range("1/1/2000", periods=5, freq="D")
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]}, index=dates, dtype="int32")

    def test_default(self):
        X_out_exp = pd.DataFrame(index=self.dates, columns=[])
        fe = SimpleFeatureEngineer()
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, X_out_exp)

    def test_keep_original(self):
        fe = SimpleFeatureEngineer(keep_original=True)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, self.X)

    def test_lag_features(self):
        rolling_features = [
            ("A", "lag", 1),
            ("B", "lag", 1),
            ("A", "lag", 2),
        ]
        X_out_exp = pd.DataFrame(
            {
                "A_lag_1": [np.nan, 1, 2, 3, 4],
                "B_lag_1": [np.nan, 3, 4, 5, 6],
                "A_lag_2": [np.nan, np.nan, 1, 2, 3],
            },
            index=self.dates,
        )
        fe = SimpleFeatureEngineer(rolling_features=rolling_features)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, X_out_exp)

    def test_rolling_features(self):
        rolling_features = [
            ("A", "mean", 2),
            ("B", "mean", 3),
            ("A", "mean", 1),
        ]
        X_out_exp = pd.DataFrame(
            {
                "A_mean_2": [np.nan, 1.5, 2.5, 3.5, 4.5],
                "B_mean_3": [np.nan, np.nan, 4, 5, 6],
                "A_mean_1": [1, 2, 3, 4, 5],
            },
            index=self.dates,
        )
        fe = SimpleFeatureEngineer(rolling_features=rolling_features)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, X_out_exp, check_dtype=False)

    def test_rolling_feature_datestring(self):
        rolling_features = [
            ("A", "mean", "2D"),
            ("B", "mean", "3D"),
            ("A", "mean", "24h"),
        ]
        X_out_exp = pd.DataFrame(
            {
                "A_mean_2D": [1, 1.5, 2.5, 3.5, 4.5],
                "B_mean_3D": [3, 3.5, 4, 5, 6],
                "A_mean_24h": [1, 2, 3, 4, 5],
            },
            index=self.dates,
        )
        fe = SimpleFeatureEngineer(rolling_features=rolling_features)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, X_out_exp, check_dtype=False)

    def test_time_features_cyclical(self):
        time_features = [
            ("day_of_year", "cyclical"),
            ("day_of_week", "cyclical"),
        ]
        X_out_exp = pd.DataFrame(
            {
                "day_of_year_cyclical_sin": [0.000, 0.017, 0.0834, 0.051, 0.069],
                "day_of_year_cyclical_cos": [1.000, 0.999, 0.999, 0.999, 0.997],
                "day_of_week_cyclical_sin": [-0.975, -0.782, 0.000, 0.782, 0.975],
                "day_of_week_cyclical_cos": [-0.222, 0.623, 1.000, 0.623, -0.223],
            },
            index=self.dates,
        )
        fe = SimpleFeatureEngineer(time_features=time_features)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(
            (X_out - X_out_exp).abs() < 5e-2,
            pd.DataFrame(
                data=np.ones(X_out.shape).astype(bool),
                index=X_out.index,
                columns=X_out.columns,
            ),
            check_dtype=False,
        )

    def test_time_features_onehot(self):
        time_features = [
            ("day_of_week", "onehot"),
        ]
        X_out_exp = pd.DataFrame(
            {
                "day_of_week_onehot_2": [0, 0, 0, 1, 0],
                "day_of_week_onehot_3": [0, 0, 0, 0, 1],
                "day_of_week_onehot_4": [0, 0, 0, 0, 0],
                "day_of_week_onehot_5": [0, 0, 0, 0, 0],
                "day_of_week_onehot_6": [1, 0, 0, 0, 0],
                "day_of_week_onehot_7": [0, 1, 0, 0, 0],
            },
            index=self.dates,
            dtype="int32",
        )

        fe = SimpleFeatureEngineer(time_features=time_features)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, X_out_exp)
