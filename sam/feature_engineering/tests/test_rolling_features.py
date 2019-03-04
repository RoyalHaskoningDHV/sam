import unittest
from pandas.testing import assert_series_equal, assert_frame_equal
from sam.feature_engineering import BuildRollingFeatures
from scipy import signal
import pandas as pd
import numpy as np


class TestRollingFeatures(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "X": [10, 12, 15, 9, 0, 0, 1]
        })

        def simple_transform(rolling_type, lookback, window_size, **kwargs):
            roller = BuildRollingFeatures(rolling_type, lookback, window_size=window_size,
                                          keep_original=False, **kwargs)
            return roller.fit_transform(self.X)
        self.simple_transform = simple_transform

    def test_lag(self):
        result = self.simple_transform('lag', 0, [0, 1, 2, 3])
        expected = pd.DataFrame({
            "X#lag_0": (10, 12, 15, 9, 0, 0, 1),
            "X#lag_1": (np.nan, 10, 12, 15, 9, 0, 0),
            "X#lag_2": (np.nan, np.nan, 10, 12, 15, 9, 0),
            "X#lag_3": (np.nan, np.nan, np.nan, 10, 12, 15, 9)
        }, columns=["X#lag_0", "X#lag_1", "X#lag_2", "X#lag_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_sum(self):
        result = self.simple_transform('sum', 0, [1, 2, 3])
        expected = pd.DataFrame({
            "X#sum_1": [10, 12, 15, 9, 0, 0, 1],
            "X#sum_2": [np.nan, 22, 27, 24, 9, 0, 1],
            "X#sum_3": [np.nan, np.nan, 37, 36, 24, 9, 1]
        }, columns=["X#sum_1", "X#sum_2", "X#sum_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_mean(self):
        result = self.simple_transform('mean', 0, [1, 2, 3])
        expected = pd.DataFrame({
            "X#mean_1": [10, 12, 15, 9, 0, 0, 1],
            "X#mean_2": [np.nan, 11, 13.5, 12, 4.5, 0, 0.5],
            "X#mean_3": [np.nan, np.nan, 12 + 1/3, 12, 8, 3, 1/3]
        }, columns=["X#mean_1", "X#mean_2", "X#mean_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_window_zero(self):
        result = self.simple_transform('lag', 1, [0, 1, 2])
        expected = pd.DataFrame({
            "X#lag_0": (np.nan, 10, 12, 15, 9, 0, 0),
            "X#lag_1": (np.nan, np.nan, 10, 12, 15, 9, 0),
            "X#lag_2": (np.nan, np.nan, np.nan, 10, 12, 15, 9)
        }, columns=["X#lag_0", "X#lag_1", "X#lag_2"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_useless_feature(self):
        # useless because no lag, it's just identity function
        result = self.simple_transform('lag', 0, 0)
        expected = pd.DataFrame({
            "X#lag_0": (10, 12, 15, 9, 0, 0, 1)
        }, columns=["X#lag_0"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_diff(self):
        result = self.simple_transform('diff', 0, [1, 2, 3])
        expected = pd.DataFrame({
            "X#diff_1": [np.nan, 2, 3, -6, -9, 0, 1],
            "X#diff_2": [np.nan, np.nan, 5, -3, -15, -9, 1],
            "X#diff_3": [np.nan, np.nan, np.nan, -1, -12, -15, -8]
        }, columns=["X#diff_1", "X#diff_2", "X#diff_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_numpos(self):
        result = self.simple_transform('numpos', 0, [1, 2, 3])
        expected = pd.DataFrame({
            "X#numpos_1": [1, 1, 1, 1, 0, 0, 1],
            "X#numpos_2": [np.nan, 2, 2, 2, 1, 0, 1],
            "X#numpos_3": [np.nan, np.nan, 3, 3, 2, 1, 1]
        }, columns=["X#numpos_1", "X#numpos_2", "X#numpos_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_ewm(self):
        result = self.simple_transform('ewm', 0, window_size=None, alpha=0.5)
        expected = pd.DataFrame({
            "X#ewm_0.5": self.X.X.ewm(alpha=0.5).mean(),
        })
        assert_frame_equal(result, expected, check_dtype=False)

        # alpha 1 should result in identity function
        result = self.simple_transform('ewm', 0, window_size=None, alpha=1)
        expected = pd.DataFrame({
            "X#ewm_1": self.X.X,
        })
        assert_frame_equal(result, expected, check_dtype=False)

    def test_fourier(self):
        # Helper function to calculate a single row of fft values
        def fastfft(values):
            return np.absolute(np.fft.fft(np.array(values)))[1:3]

        expected = [np.array([np.nan, np.nan]),
                    np.array([np.nan, np.nan]),
                    np.array([np.nan, np.nan]),
                    fastfft(self.X.X.iloc[0:4]),
                    fastfft(self.X.X.iloc[1:5]),
                    fastfft(self.X.X.iloc[2:6]),
                    fastfft(self.X.X.iloc[3:7])]
        expected = pd.DataFrame(expected,
                                columns=["X#fourier_4_1", "X#fourier_4_2"])
        result = self.simple_transform('fourier', 0, 4)
        assert_frame_equal(result, expected)

    def test_cwt(self):
        # Helper function to calculate a single row of cwt values
        def fastcwt(values, width):
            return signal.cwt(values, signal.ricker, [width])[0]

        expected = [np.array([np.nan, np.nan, np.nan, np.nan]),
                    np.array([np.nan, np.nan, np.nan, np.nan]),
                    np.array([np.nan, np.nan, np.nan, np.nan]),
                    fastcwt(self.X.X.iloc[0:4], 2.5),
                    fastcwt(self.X.X.iloc[1:5], 2.5),
                    fastcwt(self.X.X.iloc[2:6], 2.5),
                    fastcwt(self.X.X.iloc[3:7], 2.5)]

        expected = pd.DataFrame(expected,
                                columns=["X#cwt_4_0", "X#cwt_4_1", "X#cwt_4_2", "X#cwt_4_3"])
        result = self.simple_transform("cwt", 0, 4, width=2.5)
        assert_frame_equal(result, expected)

    # all the others are not tested because they are functionally exactly identical.
    # for example: std is just  lambda arr, n: arr.rolling(n).std(), which is just
    # exactly the same as sum or mean

    # Only two tests for lookback needed, because they are all treated in the exact same way
    # only fourier is treated differently.
    def test_lookback_normal(self):

        result = self.simple_transform('lag', 2, [1, 2, 3])
        expected = pd.DataFrame({
            "X#lag_1": (np.nan, np.nan, np.nan, 10, 12, 15, 9),
            "X#lag_2": (np.nan, np.nan, np.nan, np.nan, 10, 12, 15),
            "X#lag_3": (np.nan, np.nan, np.nan, np.nan, np.nan, 10, 12)
        }, columns=["X#lag_1", "X#lag_2", "X#lag_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_lookback_fourier(self):
        # Helper function to calculate a single row of fft values
        def fastfft(values):
            return np.absolute(np.fft.fft(np.array(values)))[1:3]

        expected = [np.array([np.nan, np.nan]),
                    np.array([np.nan, np.nan]),
                    np.array([np.nan, np.nan]),
                    np.array([np.nan, np.nan]),
                    fastfft(self.X.X.iloc[0:4]),
                    fastfft(self.X.X.iloc[1:5]),
                    fastfft(self.X.X.iloc[2:6])]
        expected = pd.DataFrame(expected,
                                columns=["X#fourier_4_1", "X#fourier_4_2"])
        result = self.simple_transform('fourier', 1, 4)
        assert_frame_equal(result, expected)

    # keep_original is always treated the exact same, so only one test needed
    def test_keep_original(self):
        roller = BuildRollingFeatures(rolling_type='lag', lookback=0, window_size=[1, 2, 3],
                                      keep_original=True)
        result = roller.fit_transform(self.X)
        expected = pd.DataFrame({
            "X": self.X.X,
            "X#lag_1": (np.nan, 10, 12, 15, 9, 0, 0),
            "X#lag_2": (np.nan, np.nan, 10, 12, 15, 9, 0),
            "X#lag_3": (np.nan, np.nan, np.nan, 10, 12, 15, 9)
        }, columns=["X", "X#lag_1", "X#lag_2", "X#lag_3"])
        assert_frame_equal(result, expected, check_dtype=False)

    # Only one test needed for deviation, because they are all treated the same
    # fourier is not even allowed with deviation. We choose to test with lag because it has a
    # useful shorthand since it's basically the same as diff
    def test_deviation_subtract(self):
        result = self.simple_transform('lag', 0, [1, 2, 3], deviation="subtract")
        expected = -1 * self.simple_transform('diff', 0, [1, 2, 3])
        expected.columns = result.columns

        assert_frame_equal(result, expected)

    def test_deviation_divide(self):
        result = self.simple_transform('lag', 0, [1, 2], deviation="divide")
        expected = pd.DataFrame({
            "X#lag_1": [np.nan, 10/12, 12/15, 15/9, np.inf, np.nan, 0/1],
            "X#lag_2": [np.nan, np.nan, 10/15, 12/9, np.inf, np.inf, 0/1]
        })
        assert_frame_equal(result, expected, check_dtype=False)

    def test_calc_window_size(self):
        roller = BuildRollingFeatures(rolling_type='lag', lookback=0, freq='30 minuutjes',
                                      keep_original=False, values_roll=[1, 2, 3], unit_roll='hour')
        result = roller.fit_transform(self.X)
        expected = pd.DataFrame({
            "X#lag_1_hour": (np.nan, np.nan, 10, 12, 15, 9, 0),
            "X#lag_2_hour": (np.nan, np.nan, np.nan, np.nan, 10, 12, 15),
            "X#lag_3_hour": (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10)
        }, columns=["X#lag_1_hour", "X#lag_2_hour", "X#lag_3_hour"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_get_feature_names(self):
        roller = BuildRollingFeatures('lag', lookback=0, window_size=[1, 2, 3])
        _ = roller.fit_transform(self.X)
        result = roller.get_feature_names()
        expected = ['X', 'X#lag_1', 'X#lag_2', 'X#lag_3']
        self.assertEqual(result, expected)

    def test_incorrect_inputs(self):
        # helper function. This function should already throw exceptions if the input is incorrect
        def validate(X=self.X, **kwargs):
            roller = BuildRollingFeatures(**kwargs)
            roller.fit_transform(X)
        self.assertRaises(Exception, validate)  # No input
        self.assertRaises(ValueError, validate, lookback=-1, window_size=1)  # negative lookback
        self.assertRaises(TypeError, validate, window_size="INVALID")  # typeerror
        self.assertRaises(Exception, validate, window_size=[1, 2, None])  # runtime error
        # values_roll cannot be string
        self.assertRaises(TypeError, validate, freq='15min', values_roll='30', unit_roll='minutes')
        # unit_roll must be a string
        self.assertRaises(TypeError, validate, freq='15min', values_roll=30, unit_roll=1)
        # freq must be a string
        self.assertRaises(TypeError, validate, freq=1, values_roll=30, unit_roll='minutes')
        self.assertRaises(Exception, validate, freq='45min', values_roll=[1, 2, 3],
                          unit_roll='hour')  # does not divide
        self.assertRaises(ValueError, validate, freq='foobar', values_roll=30, unit_roll='minutes')

        # must be pandas
        self.assertRaises(Exception, validate, X=np.array([[1, 2, 3], [2, 3, 4]]), window_size=1)
        self.assertRaises(TypeError, validate, window_size=1, keep_original="yes please")
        self.assertRaises(TypeError, validate, window_size=1, rolling_type=np.mean)
        self.assertRaises(TypeError, validate, window_size=1, lookback="2")

        # width must be a positive number
        self.assertRaises(TypeError, validate, window_size=1, width="2")
        self.assertRaises(TypeError, validate, window_size=1, width=[2])
        self.assertRaises(ValueError, validate, window_size=1, width=0)

        # ewm must have alpha in (0, 1]
        self.assertRaises(Exception, validate, window_size=0, rolling_type="ewm", alpha=0)
        self.assertRaises(Exception, validate, rolling_type="ewm", alpha=3)
        self.assertRaises(Exception, validate, rolling_type="ewm", alpha=[0.1, 0.2])

        # deviation cannot be used with fourier/cwt
        self.assertRaises(Exception, validate, window_size=1, deviation="something")
        self.assertRaises(Exception, validate, window_size=1, deviation="subtract",
                          rolling_type="cwt")
        self.assertRaises(Exception, validate, window_size=1, deviation="subtract",
                          rolling_type="fourier")


if __name__ == '__main__':
    unittest.main()
