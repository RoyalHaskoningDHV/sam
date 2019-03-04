import unittest
from numpy.testing import assert_array_equal
# Below are needed for setting up tests
from sam.feature_engineering import build_timefeatures
import pandas as pd
import numpy as np


class TestBuildTimeFeatures(unittest.TestCase):

    def test_march_11(self):
        time1 = '2018/03/11 15:08:12'
        time2 = '2018/03/11 16:10:11'
        freq = '15min'
        daterange = pd.date_range(time1, time2, freq=freq)
        rangelength = daterange.size
        # starts at minute 8m ends at 10, 1 hour later, step of 15 min
        minuterange = list(range(8, 10+60, 15))

        result = build_timefeatures(time1, time2, freq)
        assert_array_equal(result.TIME.values, daterange.values)
        assert_array_equal(result.index, daterange)
        assert_array_equal(result.YEAR.values, np.repeat(2018, rangelength))
        assert_array_equal(result.MONTH.values, np.repeat(3, rangelength))
        assert_array_equal(result.QUARTER.values, np.repeat(1, rangelength))
        # 11 march is week 10: https://www.epochconverter.com/weeks/2018
        assert_array_equal(result.WEEK.values, np.repeat(10, rangelength))
        # weekdays run 0-6, and 11 march is a sunday, so 6
        assert_array_equal(result.WEEKDAY.values, np.repeat(6, rangelength))
        assert_array_equal(result.WEEKEND.values, np.repeat(True, rangelength))
        assert_array_equal(result.HOUR.values, np.array([15 + x // 60 for x in minuterange]))
        # minute rolls over at 60
        assert_array_equal(result.MINUTE.values, np.array([x % 60 for x in minuterange]))
        assert_array_equal(result.DAY_PERIOD.values, np.repeat("afternoon", rangelength))

        assert_array_equal(result.columns.values,
                           np.array(['TIME', 'YEAR', 'MONTH', 'QUARTER', 'WEEK',
                                     'WEEKDAY', 'WEEKEND', 'HOUR', 'MINUTE', 'DAY_PERIOD']))

    def test_only_day_period(self):
        time1 = '2018/03/11 15:08:12'
        time2 = '2019/08/12 15:08:11'
        freq = '7D'
        daterange = pd.date_range(time1, time2, freq=freq)
        rangelength = daterange.size

        result = build_timefeatures(time1, time2, freq, year=False,
                                    seasonal=False, weekly=False, daily=True)

        assert_array_equal(result.HOUR.values, np.repeat(15, rangelength))
        assert_array_equal(result.MINUTE.values, np.repeat(8, rangelength))
        assert_array_equal(result.DAY_PERIOD.values, np.repeat("afternoon", rangelength))

        assert_array_equal(result.columns.values,
                           np.array(['TIME', 'HOUR', 'MINUTE', 'DAY_PERIOD']))

    def test_invalid_date(self):
        # This should be valueerror,   string is normally accepted, but this one isn't correct
        time1 = 'Some invalid time'
        time2 = '2019/08/12 15:08:11'
        freq = '7D'
        self.assertRaises(ValueError, build_timefeatures, time1, time2, freq)

    def test_end_before_start(self):
        time1 = '2019/08/12 15:08:11'
        time2 = '2018/03/11 15:08:12'
        freq = '7D'
        self.assertRaises(Exception, build_timefeatures, time1, time2, freq)

    def test_invalid_freq(self):
        time1 = '2018/03/11 15:08:12'
        time2 = '2019/08/12 15:08:11'
        self.assertRaises(ValueError, build_timefeatures, time1, time2, '7 days')
        self.assertRaises(ValueError, build_timefeatures, time1, time2, 7)
        self.assertRaises(ValueError, build_timefeatures, time1, time2, ['7D'])


if __name__ == '__main__':
    unittest.main()
