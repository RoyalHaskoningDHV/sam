import unittest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sam.feature_engineering import decompose_datetime
from sam.feature_engineering.decompose_datetime import fix_cyclical_features


class TestBuildTimeFeatures(unittest.TestCase):

    def test_sine_cosine_transform(self):
        df = pd.DataFrame()
        df['hr'] = np.arange(0, 5)
        df_cyc = fix_cyclical_features(df.copy(), cols=['hr'])

        expected = pd.DataFrame(
            {'hr_sin': [0., 1., 0., -1., 0.],
             'hr_cos': [1., 0., -1., 0., 1.]},
            columns=['hr_sin', 'hr_cos'])
        assert_frame_equal(df_cyc, expected)

    def test_cyclicals(self):
        # setup test dataframe
        time1 = '2019/03/11 00:00:00'
        time2 = '2019/03/11 04:00:00'
        freq = '1h'
        daterange = pd.date_range(time1, time2, freq=freq)
        test_dataframe = pd.DataFrame({'TIME': daterange, 'OTHER': 1})

        # add without cyclicals
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], [])
        expected = pd.DataFrame(
            {'TIME': daterange,
             'OTHER': 1,
             'TIME_hour': [0, 1, 2, 3, 4]},
            columns=['TIME', 'OTHER', 'TIME_hour'])
        assert_frame_equal(result, expected)

        # add cyclical test without keeping original
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], ['hour'])
        expected = pd.DataFrame(
            {'TIME': daterange,
             'OTHER': 1,
             'TIME_hour_sin': [0., 1., 0., -1., 0.],
             'TIME_hour_cos': [1., 0., -1., 0., 1.]},
            columns=['TIME', 'OTHER', 'TIME_hour_sin', 'TIME_hour_cos'])
        assert_frame_equal(result, expected)

        # and with keep original
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], ['hour'], False)
        expected = pd.DataFrame(
            {'TIME': daterange,
             'OTHER': 1,
             'TIME_hour': [0, 1, 2, 3, 4],
             'TIME_hour_sin': [0., 1., 0., -1., 0.],
             'TIME_hour_cos': [1., 0., -1., 0., 1.]},
            columns=['TIME', 'OTHER', 'TIME_hour', 'TIME_hour_sin', 'TIME_hour_cos'])
        assert_frame_equal(result, expected)

    def test_incorrect_cyclical_cols(self):
        # setup test dataframe
        time1 = '2019/03/11 00:00:00'
        time2 = '2019/03/12 00:00:00'
        freq = '2h'
        daterange = pd.date_range(time1, time2, freq=freq)
        test_dataframe = pd.DataFrame({'TIME': daterange, 'OTHER': 1})

        # this raises an error because with input columns 'TIME',
        # and requested cyclical conversion 'hour',
        # the adde column is 'TIME_hour', and not 'TIME_month'
        self.assertRaises(Exception,
                          decompose_datetime, test_dataframe, "TIME", ['hour'], ['TIME_month'])

    def test_march_11(self):
        time1 = '2018/03/11 15:08:12'
        time2 = '2018/03/11 16:10:11'
        freq = '15min'
        daterange = pd.date_range(time1, time2, freq=freq)
        rangelength = daterange.size
        # starts at minute 8m ends at 10, 1 hour later, step of 15 min
        minuterange = list(range(8, 10+60, 15))

        test_dataframe = pd.DataFrame({'TIME': daterange, 'OTHER': 1},
                                      columns=['TIME', 'OTHER'])
        components = ['year', 'month', 'quarter', 'week', 'weekday', 'hour', 'minute']

        result = decompose_datetime(test_dataframe, "TIME", components)
        assert_array_equal(result.TIME.values, daterange.values)
        assert_array_equal(result.index, test_dataframe.index)
        assert_array_equal(result.TIME_year.values, np.repeat(2018, rangelength))
        assert_array_equal(result.TIME_month.values, np.repeat(3, rangelength))
        assert_array_equal(result.TIME_quarter.values, np.repeat(1, rangelength))
        # 11 march is week 10: https://www.epochconverter.com/weeks/2018
        assert_array_equal(result.TIME_week.values, np.repeat(10, rangelength))
        # weekdays run 0-6, and 11 march is a sunday, so 6
        assert_array_equal(result.TIME_weekday.values, np.repeat(6, rangelength))
        assert_array_equal(result.TIME_hour.values, np.array([15 + x // 60 for x in minuterange]))
        # minute rolls over at 60
        assert_array_equal(result.TIME_minute.values, np.array([x % 60 for x in minuterange]))

        assert_array_equal(result.columns.values,
                           np.array(['TIME', 'OTHER', 'TIME_year', 'TIME_month', 'TIME_quarter',
                                     'TIME_week', 'TIME_weekday', 'TIME_hour', 'TIME_minute']))

    def test_no_components(self):
        time1 = '2018/03/11 15:08:12'
        time2 = '2018/03/11 16:10:11'
        freq = '15min'
        daterange = pd.date_range(time1, time2, freq=freq)

        test_dataframe = pd.DataFrame({"TIME": daterange, "OTHER": 1})

        result = decompose_datetime(test_dataframe, "TIME", [])
        assert_frame_equal(result, test_dataframe)

    def test_incorrect_column(self):
        test_dataframe = pd.DataFrame({"TIME": [1, 2, 3], "OTHER": 1})
        # Raises AttributeError because float doesn't have .dt.hour
        self.assertRaises(AttributeError, decompose_datetime, test_dataframe, "TIME", ["hour"])

    def test_absent_column(self):
        test_dataframe = pd.DataFrame({"TIME": [1, 2, 3], "OTHER": 1})
        self.assertRaises(KeyError, decompose_datetime, test_dataframe, "TEST", ["hour"])


if __name__ == '__main__':
    unittest.main()
