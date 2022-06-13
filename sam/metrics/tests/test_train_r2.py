import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
from sam.metrics import train_mean_r2, train_r2

PARAM_LIST = [
    lambda train_data, test_n: np.nanmean(train_data),
    lambda train_data, test_n: np.nanmedian(train_data),
    lambda train_data, test_n: np.interp(
        np.arange(test_n),
        np.arange(len(train_data)),
        np.convolve(train_data, np.ones(2), mode="same"),
    ),
]


class TestTrainR2(unittest.TestCase):
    def test_train_r2(self):
        from sam.metrics import train_r2
        from sklearn.metrics import r2_score

        for count, benchmark in enumerate(PARAM_LIST):
            with self.subTest():
                np.random.seed(42)
                N = 1000
                model = np.zeros(N)
                for f in [0.001, 0.005, 0.01, 0.05]:
                    for p in [0, np.pi / 4]:
                        model += (1 / f) * np.sin(2 * np.pi * f * np.arange(N) + p)
                data = model + np.random.normal(scale=250, size=N)

                custom_r2s = []
                keras_r2s = []
                for test_ratio in [0.8, 0.5, 0.2]:

                    test_n = int(test_ratio * N)
                    train_n = N - test_n
                    train_data = data[:train_n]
                    test_data = data[train_n:]
                    # train_pred = model[:train_n]
                    test_pred = model[train_n:]

                    keras_r2s.append(r2_score(test_data, test_pred))
                    custom_r2s.append(
                        train_r2(test_data, test_pred, benchmark(train_data, test_n))
                    )

                # keras r2 should decrease with decreasing test size, custom r2 should do so less
                assert_array_almost_equal(keras_r2s, [0.962522, 0.894608, 0.734713])
                if count == 0:
                    assert_array_almost_equal(custom_r2s, [0.987602, 0.989454, 0.870157])
                elif count == 1:
                    assert_array_almost_equal(custom_r2s, [0.987653, 0.990700, 0.938786])
                else:
                    assert_array_almost_equal(custom_r2s, [0.996578, 0.996330, 0.994752])

    def test_train_r2_shapes(self):
        # the function cannot handle data with multiple dimensions. It does however ravel
        # empty dimensions (x, 1).
        with self.assertRaises(ValueError):
            train_r2(np.random.random(size=(12, 2)), np.random.random(size=(12, 2)), 0)
        with self.assertRaises(ValueError):
            train_r2(np.random.random(size=(12, 1)), np.random.random(size=(12, 2)), 0)
        with self.assertRaises(ValueError):
            train_r2(np.random.random(size=(12, 2)), np.random.random(size=(12, 1)), 0)
        with self.assertRaises(ValueError):
            train_r2(
                np.random.random(size=(12, 1)),
                np.random.random(size=(12, 1)),
                np.random.random(size=(12, 2)),
            )
        # benchmark array has to be same size as true array
        with self.assertRaises(ValueError):
            train_r2(
                np.random.random(size=(12, 1)),
                np.random.random(size=(12, 1)),
                np.random.random(size=(13, 1)),
            )

    def test_train_mean_r2_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered
            warnings.simplefilter("always")

            # Prepare parameters
            true_array = np.array([0, 1, 2, 3, 4])
            predicted_array = np.array([1, 2, 3, 4, 5])

            # Trigger a warning
            r2 = train_mean_r2(true_array, predicted_array, np.nanmedian(true_array))

            # Verify some things
            if r2 != 0.5:
                raise AssertionError("r2 is not 0.5")
            if len(w) != 1:
                raise AssertionError("len(w) is not 1")
            if not issubclass(w[-1].category, DeprecationWarning):
                raise AssertionError("Warning is not DeprecationWarning")
            if "DEPRECATED" not in str(w[-1].message):
                raise AssertionError("Warning message does not contain DEPRECATED")


if __name__ == "__main__":
    unittest.main()
