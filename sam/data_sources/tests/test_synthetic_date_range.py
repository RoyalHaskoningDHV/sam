import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sam.data_sources import synthetic_date_range


class TestCreateSyntheticTimes(unittest.TestCase):
    def test_nonoise(self):
        result = synthetic_date_range(start="2016-01-01", end="2016-01-01 03:00:00", freq="H")
        expected = pd.DatetimeIndex(
            np.array(["2016-01-01 00:00:00", "2016-01-01 01:00:00", "2016-01-01 02:00:00"])
        )
        assert_array_equal(result, expected)

    def test_shortseries(self):
        result = synthetic_date_range(start="2016-01-01", end="2016-01-01 01:00:00", freq="H")
        expected = pd.DatetimeIndex(np.array(["2016-01-01 00:00:00"]))
        assert_array_equal(result, expected)

    def test_emptyseries(self):
        result = synthetic_date_range(start="2016-01-01", end="2016-01-01 00:30:00", freq="2H")
        expected = pd.DatetimeIndex(np.array(["2016-01-01 00:00:00"]))
        assert_array_equal(result, expected)

    def test_incorrect_input(self):
        self.assertRaises(
            Exception,
            synthetic_date_range,
            "2016-01-01 02:00:00",
            "2016-01-01 00:30:00",
            "2H",
        )
        self.assertRaises(Exception, synthetic_date_range, "2016-01-01", "2017-01-01", "1 hour")
        self.assertRaises(
            Exception, synthetic_date_range, "2016-01-01", "2017-01-01", max_delay="1"
        )
        self.assertRaises(
            Exception,
            synthetic_date_range,
            "2016-01-01",
            "2017-01-01",
            random_stop_freq=[1],
        )
        # negative random stop freq is allowed, is just treated as 0. More than 1 is not
        self.assertRaises(
            Exception,
            synthetic_date_range,
            "2016-01-01",
            "2017-01-01",
            random_stop_freq=2,
        )
        # random stop max length is only checked if there is at least 1 random stop
        self.assertRaises(
            Exception,
            synthetic_date_range,
            "2016-01-01",
            "2017-01-01",
            random_stop_freq=1,
            random_stop_max_length=-1,
        )

    def test_delays(self):
        result = synthetic_date_range(
            start="2016-01-01 00:00:00",
            end="2017-01-01 00:30:00",
            freq="H",
            max_delay=600,
        )
        # Delays should be between 1 hour and 1H10M (1 hour normal, 0-10 min delay)
        self.assertGreater((result - result.to_series().shift())[1:].min(), pd.Timedelta("60min"))
        self.assertLess((result - result.to_series().shift())[1:].max(), pd.Timedelta("70min"))
        # These stricter bounds are probabilistic, but the probability of failure is approximately
        # the same probability as being hit by lightning 43 consecutive days. (probability for any
        # single day is 1 in 100 million)
        self.assertLess((result - result.to_series().shift())[1:].min(), pd.Timedelta("61min"))
        self.assertGreater((result - result.to_series().shift())[1:].max(), pd.Timedelta("69min"))

    def test_random_stops(self):
        # We test only one month instead of one year, because this option has a for-loop and is
        # therefore unfortunately a bit slower
        result = synthetic_date_range(
            start="2016-01-01 00:00:00",
            end="2016-02-01 00:30:00",
            freq="H",
            random_stop_freq=0.5,
        )
        # approx half of all points should have been removed.
        expected_length = 31 * 24
        # probability of failure is ~10^-30 for each of these 2 tests
        # https://www.wolframalpha.com/input/?i=probability+of+less+than+223+heads+in+744+coin+tosses
        self.assertGreater(result.size, (0.3 * expected_length))
        self.assertLess(result.size, (0.7 * expected_length))

    def test_longer_stops(self):
        result = synthetic_date_range(
            start="2016-01-01 00:00:00",
            end="2016-02-01 00:30:00",
            freq="H",
            random_stop_freq=0.1,
            random_stop_max_length=3,
        )
        expected_length = 31 * 24
        # naively, 0.1*3 = 30% of all observations are taken out, but gaps can overlap.
        # roughly, the number of observations that were removed should be between 10% and 30%
        self.assertGreater(result.size, (0.7 * expected_length))
        self.assertLess(result.size, (0.9 * expected_length))

    def test_seed(self):
        foo = synthetic_date_range(
            start="2016-01-01 00:00:00",
            end="2016-02-01 00:30:00",
            freq="H",
            max_delay=100,
            random_stop_freq=0.1,
            random_stop_max_length=3,
            seed=42,
        )
        bar = synthetic_date_range(
            start="2016-01-01 00:00:00",
            end="2016-02-01 00:30:00",
            freq="H",
            max_delay=100,
            random_stop_freq=0.1,
            random_stop_max_length=3,
            seed=42,
        )
        assert_array_equal(foo, bar)


if __name__ == "__main__":
    unittest.main()
