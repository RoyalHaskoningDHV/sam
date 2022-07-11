import unittest

import nfft
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sam.feature_engineering import BuildRollingFeatures
from scipy import signal


class TestRollingFeatures(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"X": [10, 12, 15, 9, 0, 0, 1]})

        self.times = [
            "2011-01-01 00:00",
            "2011-01-01 01:00",
            "2011-01-01 02:00",
            "2011-01-01 04:00",
            "2011-01-01 05:00",
            "2011-01-01 06:00",
            "2011-01-01 07:00",
        ]
        self.X_times = pd.DataFrame(
            {"X": [10, 12, 15, 9, 0, 0, 1]}, index=pd.DatetimeIndex(self.times)
        )

        def simple_transform(rolling_type, lookback, window_size, **kwargs):
            roller = BuildRollingFeatures(
                rolling_type, lookback, window_size=window_size, keep_original=False, **kwargs
            )
            return roller.fit_transform(self.X)

        self.simple_transform = simple_transform

    def test_lag(self):
        result = self.simple_transform("lag", 0, [0, 1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#lag_0": (10, 12, 15, 9, 0, 0, 1),
                "X#lag_1": (np.nan, 10, 12, 15, 9, 0, 0),
                "X#lag_2": (np.nan, np.nan, 10, 12, 15, 9, 0),
                "X#lag_3": (np.nan, np.nan, np.nan, 10, 12, 15, 9),
            },
            columns=["X#lag_0", "X#lag_1", "X#lag_2", "X#lag_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_sum(self):
        result = self.simple_transform("sum", 0, [1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#sum_1": [10, 12, 15, 9, 0, 0, 1],
                "X#sum_2": [np.nan, 22, 27, 24, 9, 0, 1],
                "X#sum_3": [np.nan, np.nan, 37, 36, 24, 9, 1],
            },
            columns=["X#sum_1", "X#sum_2", "X#sum_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_mean(self):
        result = self.simple_transform("mean", 0, [1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#mean_1": [10, 12, 15, 9, 0, 0, 1],
                "X#mean_2": [np.nan, 11, 13.5, 12, 4.5, 0, 0.5],
                "X#mean_3": [np.nan, np.nan, 12 + 1 / 3, 12, 8, 3, 1 / 3],
            },
            columns=["X#mean_1", "X#mean_2", "X#mean_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_trimmean(self):
        result = self.simple_transform("trimmean", 0, [3, 4, 5], proportiontocut=0.3)
        expected = pd.DataFrame(
            {
                "X#trimmean_3": [np.nan, np.nan, 12 + 1 / 3, 12, 8, 3, 1 / 3],
                "X#trimmean_4": [np.nan, np.nan, np.nan, 11, 10.5, 4.5, 0.5],
                "X#trimmean_5": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    10 + 1 / 3,
                    7,
                    3 + 1 / 3,
                ],
            },
            columns=["X#trimmean_3", "X#trimmean_4", "X#trimmean_5"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_window_zero(self):
        result = self.simple_transform("lag", 1, [0, 1, 2])
        expected = pd.DataFrame(
            {
                "X#lag_0": (np.nan, 10, 12, 15, 9, 0, 0),
                "X#lag_1": (np.nan, np.nan, 10, 12, 15, 9, 0),
                "X#lag_2": (np.nan, np.nan, np.nan, 10, 12, 15, 9),
            },
            columns=["X#lag_0", "X#lag_1", "X#lag_2"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_useless_feature(self):
        # useless because no lag, it's just identity function
        result = self.simple_transform("lag", 0, 0)
        expected = pd.DataFrame({"X#lag_0": (10, 12, 15, 9, 0, 0, 1)}, columns=["X#lag_0"])
        assert_frame_equal(result, expected, check_dtype=False)

    def test_diff(self):
        result = self.simple_transform("diff", 0, [1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#diff_1": [np.nan, 2, 3, -6, -9, 0, 1],
                "X#diff_2": [np.nan, np.nan, 5, -3, -15, -9, 1],
                "X#diff_3": [np.nan, np.nan, np.nan, -1, -12, -15, -8],
            },
            columns=["X#diff_1", "X#diff_2", "X#diff_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_numpos(self):
        result = self.simple_transform("numpos", 0, [1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#numpos_1": [1, 1, 1, 1, 0, 0, 1],
                "X#numpos_2": [np.nan, 2, 2, 2, 1, 0, 1],
                "X#numpos_3": [np.nan, np.nan, 3, 3, 2, 1, 1],
            },
            columns=["X#numpos_1", "X#numpos_2", "X#numpos_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_ewm(self):
        result = self.simple_transform("ewm", 0, window_size=None, alpha=0.5)
        expected = pd.DataFrame(
            {
                "X#ewm_0.5": self.X.X.ewm(alpha=0.5).mean(),
            }
        )
        assert_frame_equal(result, expected, check_dtype=False)

        # alpha 1 should result in identity function
        result = self.simple_transform("ewm", 0, window_size=None, alpha=1)
        expected = pd.DataFrame(
            {
                "X#ewm_1": self.X.X,
            }
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_fourier(self):
        # Helper function to calculate a single row of fft values
        def fastfft(values):
            return np.absolute(np.fft.fft(np.array(values)))[1:3]

        expected = [
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            fastfft(self.X.X.iloc[0:4]),
            fastfft(self.X.X.iloc[1:5]),
            fastfft(self.X.X.iloc[2:6]),
            fastfft(self.X.X.iloc[3:7]),
        ]
        expected = pd.DataFrame(expected, columns=["X#fourier_4_1", "X#fourier_4_2"])
        result = self.simple_transform("fourier", 0, 4)
        assert_frame_equal(result, expected)

    def test_cwt(self):
        # Helper function to calculate a single row of cwt values
        def fastcwt(values, width):
            return signal.cwt(values, signal.ricker, [width])[0]

        expected = [
            np.array([np.nan, np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan, np.nan]),
            fastcwt(self.X.X.iloc[0:4], 2.5),
            fastcwt(self.X.X.iloc[1:5], 2.5),
            fastcwt(self.X.X.iloc[2:6], 2.5),
            fastcwt(self.X.X.iloc[3:7], 2.5),
        ]

        expected = pd.DataFrame(
            expected, columns=["X#cwt_4_0", "X#cwt_4_1", "X#cwt_4_2", "X#cwt_4_3"]
        )
        result = self.simple_transform("cwt", 0, 4, width=2.5)
        assert_frame_equal(result, expected)

    def test_withmissing(self):
        X = pd.DataFrame({"X": [10, 12, 15, np.nan, 0, 0, 1]})
        roller = BuildRollingFeatures("sum", lookback=0, window_size=2, keep_original=False)
        result = roller.fit_transform(X)
        expected = pd.DataFrame({"X#sum_2": [np.nan, 22, 27, np.nan, np.nan, 0, 1]})
        assert_frame_equal(result, expected)

    def test_fourier_withmissing(self):
        def fastfft(values):
            return np.absolute(np.fft.fft(np.array(values)))[1]

        # Tests T502: missing values caused the results to shift.
        X = pd.DataFrame({"X": [10, 12, 15, np.nan, 0, 0, 1]})

        roller = BuildRollingFeatures("fourier", lookback=0, window_size=2, keep_original=False)
        result = roller.fit_transform(X)

        expected = [
            np.nan,
            fastfft(X.X.iloc[0:2]),
            fastfft(X.X.iloc[1:3]),
            np.nan,
            np.nan,
            fastfft(X.X.iloc[4:6]),
            fastfft(X.X.iloc[5:7]),
        ]
        expected = pd.DataFrame({"X#fourier_2_1": expected})

        assert_frame_equal(result, expected)

    # all the others are not tested because they are functionally exactly identical.
    # for example: std is just  lambda arr, n: arr.rolling(n).std(), which is just
    # exactly the same as sum or mean

    # Only two tests for lookback needed, because they are all treated in the exact same way
    # only fourier is treated differently.
    def test_lookback_normal(self):

        result = self.simple_transform("lag", 2, [1, 2, 3])
        expected = pd.DataFrame(
            {
                "X#lag_1": (np.nan, np.nan, np.nan, 10, 12, 15, 9),
                "X#lag_2": (np.nan, np.nan, np.nan, np.nan, 10, 12, 15),
                "X#lag_3": (np.nan, np.nan, np.nan, np.nan, np.nan, 10, 12),
            },
            columns=["X#lag_1", "X#lag_2", "X#lag_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_lookback_fourier(self):
        # Helper function to calculate a single row of fft values
        def fastfft(values):
            return np.absolute(np.fft.fft(np.array(values)))[1:3]

        expected = [
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            fastfft(self.X.X.iloc[0:4]),
            fastfft(self.X.X.iloc[1:5]),
            fastfft(self.X.X.iloc[2:6]),
        ]
        expected = pd.DataFrame(expected, columns=["X#fourier_4_1", "X#fourier_4_2"])
        result = self.simple_transform("fourier", 1, 4)
        assert_frame_equal(result, expected)

    # keep_original is always treated the exact same, so only one test needed
    def test_keep_original(self):
        roller = BuildRollingFeatures(
            rolling_type="lag", lookback=0, window_size=[1, 2, 3], keep_original=True
        )
        result = roller.fit_transform(self.X)
        expected = pd.DataFrame(
            {
                "X": self.X.X,
                "X#lag_1": (np.nan, 10, 12, 15, 9, 0, 0),
                "X#lag_2": (np.nan, np.nan, 10, 12, 15, 9, 0),
                "X#lag_3": (np.nan, np.nan, np.nan, 10, 12, 15, 9),
            },
            columns=["X", "X#lag_1", "X#lag_2", "X#lag_3"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    # Only one test needed for deviation, because they are all treated the same
    # fourier is not even allowed with deviation. We choose to test with lag because it has a
    # useful shorthand since it's basically the same as diff
    def test_deviation_subtract(self):
        result = self.simple_transform("lag", 0, [1, 2, 3], deviation="subtract")
        expected = -1 * self.simple_transform("diff", 0, [1, 2, 3])
        expected.columns = result.columns

        assert_frame_equal(result, expected)

    def test_deviation_divide(self):
        result = self.simple_transform("lag", 0, [1, 2], deviation="divide")
        expected = pd.DataFrame(
            {
                "X#lag_1": [np.nan, 10 / 12, 12 / 15, 15 / 9, np.inf, np.nan, 0 / 1],
                "X#lag_2": [np.nan, np.nan, 10 / 15, 12 / 9, np.inf, np.inf, 0 / 1],
            }
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_get_feature_names(self):
        roller = BuildRollingFeatures("lag", lookback=0, window_size=[1, 2, 3])
        _ = roller.fit_transform(self.X)
        result = roller.get_feature_names()
        expected = ["X", "X#lag_1", "X#lag_2", "X#lag_3"]
        self.assertEqual(result, expected)

    def test_get_feature_names_with_lookback(self):
        roller = BuildRollingFeatures(
            "lag", lookback=0, window_size=[1, 2, 3], add_lookback_to_colname=True
        )
        _ = roller.fit_transform(self.X)
        result = roller.get_feature_names()
        expected = [
            "X",
            "X#lag_1_lookback_0",
            "X#lag_2_lookback_0",
            "X#lag_3_lookback_0",
        ]
        self.assertEqual(result, expected)

        roller = BuildRollingFeatures(
            "lag", lookback=2, window_size=[1, 2, 3], add_lookback_to_colname=True
        )
        _ = roller.fit_transform(self.X)
        result = roller.get_feature_names()
        expected = [
            "X",
            "X#lag_1_lookback_2",
            "X#lag_2_lookback_2",
            "X#lag_3_lookback_2",
        ]
        self.assertEqual(result, expected)

    def test_datetimeindex(self):
        roller = BuildRollingFeatures("sum", lookback=1, window_size=["61min", "3H"])
        result = roller.fit_transform(self.X_times)
        expected = pd.DataFrame(
            {
                "X": [10, 12, 15, 9, 0, 0, 1],
                "X#sum_61min": [np.nan, 10, 22, 27, 9, 9, 0],
                "X#sum_3H": [np.nan, 10, 22, 37, 24, 9, 9],
            },
            columns=["X", "X#sum_61min", "X#sum_3H"],
            index=pd.DatetimeIndex(self.times),
        )

        assert_frame_equal(result, expected)

    def test_timecol(self):
        X = self.X.copy()
        X["TIME"] = self.times
        roller = BuildRollingFeatures(
            "sum", lookback=1, window_size=["61min", "3H"], timecol="TIME"
        )
        result = roller.fit_transform(X)
        expected = pd.DataFrame(
            {
                "X": [10, 12, 15, 9, 0, 0, 1],
                "X#sum_61min": [np.nan, 10, 22, 27, 9, 9, 0],
                "X#sum_3H": [np.nan, 10, 22, 37, 24, 9, 9],
            },
            columns=["X", "X#sum_61min", "X#sum_3H"],
        )

        assert_frame_equal(result, expected)

    def test_datetimeindex_single_window(self):
        # Checks bug from T592
        roller = BuildRollingFeatures("sum", lookback=1, window_size="61min")
        result = roller.fit_transform(self.X_times)
        expected = pd.DataFrame(
            {
                "X": [10, 12, 15, 9, 0, 0, 1],
                "X#sum_61min": [np.nan, 10, 22, 27, 9, 9, 0],
            },
            columns=["X", "X#sum_61min"],
            index=pd.DatetimeIndex(self.times),
        )

        assert_frame_equal(result, expected)

    def test_nfft(self):
        X = pd.DataFrame({"X": pd.concat([self.X.X] * 2)})  # length 14
        times = pd.Series(pd.to_datetime(self.times))
        X["TIME"] = pd.concat([times, times + pd.Timedelta("8H")])

        roller = BuildRollingFeatures(
            "nfft",
            lookback=0,
            window_size=["500min"],
            timecol="TIME",
            nfft_ncol=3,
            keep_original=False,
        )
        result = roller.fit_transform(X)

        def nfft_helper(times, values):
            # The minimum to copy from rolling_features that would be super annoying to do by hand
            times = times / np.max(times) - 0.5  # Scale between -0.5 and 0.5
            f = nfft.nfft(times, values - np.mean(values))
            return pd.Series(np.abs(np.fft.fftshift(f))[1:4])

        expected = pd.concat(
            [
                pd.Series([0, 0, 0]),  # nfft needs at least 6 values so these will just be 0
                pd.Series([0, 0, 0]),
                pd.Series([0, 0, 0]),
                pd.Series([0, 0, 0]),
                pd.Series([0, 0, 0]),
                nfft_helper(np.array([0, 1, 2, 4, 5, 6]), np.array([10, 12, 15, 9, 0, 0])),
                nfft_helper(np.array([1, 2, 4, 5, 6, 7]) - 1, np.array([12, 15, 9, 0, 0, 1])),
                nfft_helper(
                    np.array([0, 1, 2, 4, 5, 6, 7, 8]),
                    np.array([10, 12, 15, 9, 0, 0, 1, 10]),
                ),
                nfft_helper(
                    np.array([1, 2, 4, 5, 6, 7, 8, 9]) - 1,
                    np.array([12, 15, 9, 0, 0, 1, 10, 12]),
                ),
                nfft_helper(
                    np.array([2, 4, 5, 6, 7, 8, 9, 10]) - 2,
                    np.array([15, 9, 0, 0, 1, 10, 12, 15]),
                ),
                nfft_helper(
                    np.array([4, 5, 6, 7, 8, 9, 10, 12]) - 4,
                    np.array([9, 0, 0, 1, 10, 12, 15, 9]),
                ),
                nfft_helper(
                    np.array([5, 6, 7, 8, 9, 10, 12, 13]) - 5,
                    np.array([0, 0, 1, 10, 12, 15, 9, 0]),
                ),
                nfft_helper(
                    np.array([6, 7, 8, 9, 10, 12, 13, 14]) - 6,
                    np.array([0, 1, 10, 12, 15, 9, 0, 0]),
                ),
                nfft_helper(
                    np.array([7, 8, 9, 10, 12, 13, 14, 15]) - 7,
                    np.array([1, 10, 12, 15, 9, 0, 0, 1]),
                ),
            ],
            axis=1,
        ).transpose()
        expected.index = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
        expected.columns = ["X#nfft_500min_0", "X#nfft_500min_1", "X#nfft_500min_2"]
        # When window_size is 6, only 2 values are calculated, so remove the third value
        expected["X#nfft_500min_2"].iloc[5:7] = 0

        assert_frame_equal(result, expected)

    def test_incorrect_inputs(self):
        # helper function. This function should already throw exceptions if the input is incorrect
        def validate(X=self.X, **kwargs):
            roller = BuildRollingFeatures(**kwargs)
            roller.fit_transform(X)

        self.assertRaises(Exception, validate)  # No input
        self.assertRaises(ValueError, validate, lookback=-1, window_size=1)  # negative lookback
        self.assertRaises(Exception, validate, window_size="INVALID")
        self.assertRaises(Exception, validate, window_size=[1, 2, None])  # runtime error

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
        self.assertRaises(
            Exception, validate, window_size=1, deviation="subtract", rolling_type="cwt"
        )
        self.assertRaises(
            Exception,
            validate,
            window_size=1,
            deviation="subtract",
            rolling_type="fourier",
        )

        # trimmean must have proportiontocut in [0, 0.5)
        self.assertRaises(
            Exception,
            validate,
            window_size=0,
            rolling_type="trimmean",
            proportiontocut=-0.2,
        )
        self.assertRaises(Exception, validate, rolling_type="trimmean", proportiontocut=0.8)
        self.assertRaises(Exception, validate, rolling_type="trimmean", proportiontocut=[0.1, 0.2])

        # timeoffset can only be used with datetimeindex, and not with lag/ewm/fourier/diff
        self.assertRaises(ValueError, validate, window_size="1H")
        self.assertRaises(
            ValueError, validate, X=self.X_times, window_size="1H", rolling_type="lag"
        )
        self.assertRaises(
            ValueError,
            validate,
            X=self.X_times,
            window_size=[1, "1H"],
            rolling_type="lag",
        )


if __name__ == "__main__":
    unittest.main()
