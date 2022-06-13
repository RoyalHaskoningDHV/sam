import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sam.data_sources import synthetic_timeseries


class TestSyntheticTimeseries(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2015-01-01", "2015-01-01 03:00:00", freq="H").to_series()
        self.many_dates = pd.date_range("2015-01-01", "2015-02-01 00:00:00", freq="H").to_series()

    def test_nonoise(self):
        result = synthetic_timeseries(self.dates)
        expected = np.array([0, 0, 0, 0], dtype=np.float64)
        assert_array_equal(result, expected)

    def test_some_incorrect_inputs(self):
        self.assertRaises(Exception, synthetic_timeseries, self.dates, minmax_values=0)
        self.assertRaises(Exception, synthetic_timeseries, self.dates, cutoff_values=True)
        self.assertRaises(Exception, synthetic_timeseries, self.dates, monthly=None)
        self.assertRaises(Exception, synthetic_timeseries, self.dates, daily="1")
        self.assertRaises(Exception, synthetic_timeseries, self.dates, hourly=["1"])
        self.assertRaises(Exception, synthetic_timeseries, self.dates, monthnoise=1)
        self.assertRaises(Exception, synthetic_timeseries, self.dates, daynoise=("normal", "a"))
        self.assertRaises(Exception, synthetic_timeseries, self.dates, noise=1)
        self.assertRaises(Exception, synthetic_timeseries, self.dates, negabs="1")
        self.assertRaises(Exception, synthetic_timeseries, self.dates, random_missing=2)
        # There must be at least 2 datetimes to create a spline
        empty_dates = pd.Series([], dtype="datetime64[ns]")
        self.assertRaises(Exception, synthetic_timeseries, empty_dates)
        empty_dates = pd.Series(["2016-01-01 00:00:00"], dtype="datetime64[ns]")
        self.assertRaises(Exception, synthetic_timeseries, empty_dates)

    def test_noise(self):
        without_noise = synthetic_timeseries(self.many_dates, 1, 2, 3)
        with_noise = synthetic_timeseries(
            self.many_dates, 1, 2, 3, noise={"normal": 2, "poisson": 1}
        )
        self.assertGreater(np.var(with_noise), np.var(without_noise))

    def test_minmax(self):
        result = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            noise={"normal": 2, "poisson": 1},
            minmax_values=(0, 10),
        )
        # Result should be between 0 and 10, with very little margin for error
        self.assertGreater(np.nanmax(result), 9.999)
        self.assertLess(np.nanmin(result), 0.001)

    def test_cutoff(self):
        result = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            noise={"normal": 2, "poisson": 1},
            minmax_values=(0, 10),
            cutoff_values=(2, 8),
        )
        # There is slightly more margin for error, but values should be between 2 and 8 now
        self.assertGreater(np.nanmax(result), 7.5)
        self.assertLess(np.nanmin(result), 2.5)

    def test_negabs(self):
        # negabs makes sure the result is more centered around 0, so the average should be lower
        without_negabs = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            noise={"normal": 2, "poisson": 1},
            minmax_values=(0, 10),
        )
        with_negabs = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            noise={"normal": 2, "poisson": 1},
            minmax_values=(0, 10),
            negabs=5,
        )
        self.assertGreater(np.nanmean(without_negabs), np.nanmean(with_negabs))

    def test_random_missing(self):
        result = synthetic_timeseries(self.many_dates, 1, 2, 3, random_missing=0.5)
        expected = self.many_dates.size
        # We would expect between 0.3 and 0.7 of observations to be nan.
        # This test can probabilistically fail, but the chance of that is 1 in billions of billions
        number_of_nans = np.count_nonzero(np.isnan(result))
        self.assertGreater(number_of_nans, 0.3 * expected)
        self.assertLess(number_of_nans, 0.7 * expected)

    def test_seed(self):
        # result must be consistent if using the same seed
        foo = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            ("normal", 4),
            ("poisson", 3),
            {"normal": 2, "poisson": 3},
            seed=42,
        )
        bar = synthetic_timeseries(
            self.many_dates,
            1,
            2,
            3,
            ("normal", 4),
            ("poisson", 3),
            {"normal": 2, "poisson": 3},
            seed=42,
        )
        assert_array_equal(foo, bar)


if __name__ == "__main__":
    unittest.main()
