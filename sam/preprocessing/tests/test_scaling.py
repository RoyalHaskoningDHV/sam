import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from sam.preprocessing import scale_train_test


class TestScaling(unittest.TestCase):
    def test_standard_scaler_series(self):

        N = 10
        basex = np.random.random(N)
        basey = np.random.random(N)
        X_base = pd.DataFrame([basex, basex]).astype("float").T
        y_base = pd.Series(basey).astype("float")
        X_train, X_test, y_train, y_test = X_base, X_base, y_base, y_base
        X_train_s, X_test_s, y_train_s, y_test_s, X_scaler, y_scaler = scale_train_test(
            X_train, X_test, y_train, y_test
        )

        # test outcome shapes
        self.assertEqual(X_train.shape, X_train_s.shape)
        self.assertEqual(X_test.shape, X_test_s.shape)
        self.assertEqual(y_test.shape, y_test_s.shape)
        self.assertEqual(y_train.shape, y_train_s.shape)

        # test whether mean is really close to 0
        assert_series_equal(X_train_s.mean(), pd.Series([0.0, 0.0]))
        assert_series_equal(X_test_s.mean(), pd.Series([0.0, 0.0]))
        self.assertAlmostEqual(y_test_s.mean(), 0)
        self.assertAlmostEqual(y_train_s.mean(), 0)

        # test whether std is really 1
        assert_series_equal(np.std(X_train_s), pd.Series([1.0, 1.0]))
        assert_series_equal(np.std(X_test_s), pd.Series([1.0, 1.0]))
        self.assertAlmostEqual(np.std(y_test_s), 1)
        self.assertAlmostEqual(np.std(y_train_s), 1)

        # test whether x and y scaler were really seperately fitted on the
        # x and y data
        self.assertEqual(X_scaler.mean_.shape, (2,))
        self.assertEqual(y_scaler.mean_.shape, (1,))
        self.assertNotEqual(X_scaler.mean_[0], y_scaler.mean_[0])

    def test_standard_scaler_dataframe(self):

        N = 10
        basex = np.random.random(N)
        basey = np.random.random(N)
        X_base = pd.DataFrame([basex, basex]).astype("float").T
        y_base = pd.DataFrame([basey, basey, basey]).astype("float").T
        X_train, X_test, y_train, y_test = X_base, X_base, y_base, y_base
        X_train_s, X_test_s, y_train_s, y_test_s, X_scaler, y_scaler = scale_train_test(
            X_train, X_test, y_train, y_test
        )

        # test outcome shapes
        self.assertEqual(X_train.shape, X_train_s.shape)
        self.assertEqual(X_test.shape, X_test_s.shape)
        self.assertEqual(y_test.shape, y_test_s.shape)
        self.assertEqual(y_train.shape, y_train_s.shape)

        # test whether mean is really close to 0
        assert_series_equal(X_train_s.mean(), pd.Series([0.0, 0.0]))
        assert_series_equal(X_test_s.mean(), pd.Series([0.0, 0.0]))
        assert_series_equal(y_train_s.mean(), pd.Series([0.0, 0.0, 0.0]))
        assert_series_equal(y_test_s.mean(), pd.Series([0.0, 0.0, 0.0]))

        # test whether std is really 1
        assert_series_equal(np.std(X_train_s), pd.Series([1.0, 1.0]))
        assert_series_equal(np.std(X_test_s), pd.Series([1.0, 1.0]))
        assert_series_equal(np.std(y_train_s), pd.Series([1.0, 1.0, 1.0]))
        assert_series_equal(np.std(y_test_s), pd.Series([1.0, 1.0, 1.0]))

        # test whether x and y scaler were really seperately fitted on the
        # x and y data
        self.assertEqual(X_scaler.mean_.shape, (2,))
        self.assertEqual(y_scaler.mean_.shape, (3,))
        self.assertNotEqual(X_scaler.mean_[0], y_scaler.mean_[0])
