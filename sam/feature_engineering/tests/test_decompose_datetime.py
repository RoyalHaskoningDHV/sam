import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sam.feature_engineering import decompose_datetime
from sam.feature_engineering import recode_cyclical_features


class TestBuildTimeFeatures(unittest.TestCase):

    def test_sine_cosine_transform(self):
        df = pd.DataFrame()
        df['hr'] = np.arange(0, 5)
        # Since 'hr' is not a dt attribute, we have to provide max ourself
        df_cyc = recode_cyclical_features(df.copy(), cols=['hr'], cyclical_maxes=[4])

        expected = pd.DataFrame(
            {'hr_sin': [0., 1., 0., -1., 0.],
             'hr_cos': [1., 0., -1., 0., 1.]},
            columns=['hr_sin', 'hr_cos'])
        assert_frame_equal(df_cyc, expected)

    def test_recode_cyclicals_int32(self):
        # Checks if T606 was fixed

        df = pd.DataFrame()
        df['hr'] = np.arange(0, 5)
        df['hr'] = df['hr'].astype(np.int32)
        # Should not throw an error
        # Since 'hr' is not a dt attribute, we have to provide max ourself
        df_cyc = recode_cyclical_features(df.copy(), cols=['hr'], cyclical_maxes=[4])

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
        test_dataframe = pd.DataFrame({'TIME': daterange, 'OTHER': 1},
                                      columns=['TIME', 'OTHER'])

        # add without cyclicals
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], [])
        expected = pd.DataFrame(
            {'TIME': daterange,
             'OTHER': 1,
             'TIME_hour': [0, 1, 2, 3, 4]},
            columns=['TIME', 'OTHER', 'TIME_hour'])
        assert_frame_equal(result, expected)

        # add cyclical test without keeping original
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], ['hour'], cyclical_maxes=[4])
        expected = pd.DataFrame(
            {'TIME': daterange,
             'OTHER': 1,
             'TIME_hour_sin': [0., 1., 0., -1., 0.],
             'TIME_hour_cos': [1., 0., -1., 0., 1.]},
            columns=['TIME', 'OTHER', 'TIME_hour_sin', 'TIME_hour_cos'])
        assert_frame_equal(result, expected)

        # and with keep original
        result = decompose_datetime(test_dataframe, "TIME", ['hour'], ['hour'], cyclical_maxes=[4],
                                    remove_categorical=False)
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

    def test_remove_original(self):
        # Test if T643 was fixed

        time1 = '2018/03/11 15:08:12'
        time2 = '2018/03/11 16:10:11'
        freq = '15min'
        daterange = pd.date_range(time1, time2, freq=freq)

        data = pd.DataFrame({"TIME": daterange, "OTHER": 1})
        expected = pd.DataFrame({"TIME_minute": [8, 23, 38, 53, 8]})

        result = decompose_datetime(data, "TIME", ['minute'], keep_original=False)
        assert_frame_equal(result, expected)

        # Also try it with cyclical
        cyclical_result = decompose_datetime(data, "TIME", ['minute'], ['minute'],
                                             keep_original=False)
        assert_array_equal(cyclical_result.columns.values,
                           np.array(['TIME_minute_sin', 'TIME_minute_cos']))

        # Also try just cyclical, no decompose
        result1 = recode_cyclical_features(data, ['OTHER'], remove_categorical=True,
                                           column='', keep_original=False,
                                           cyclical_maxes=[1])
        result2 = recode_cyclical_features(data, ['OTHER'], remove_categorical=False,
                                           column='', keep_original=False,
                                           cyclical_maxes=[1])

        # The first had remove_categorical, so TIME and OTHER are both dropped
        assert_array_equal(result1.columns.values, np.array(['OTHER_sin', 'OTHER_cos']))
        # The second had remove_categorical False, so TIME is dropped but OTHER isn't
        assert_array_equal(result2.columns.values, np.array(['OTHER', 'OTHER_sin', 'OTHER_cos']))

    def test_incorrect_column(self):
        test_dataframe = pd.DataFrame({"TIME": [1, 2, 3], "OTHER": 1})
        # Raises AttributeError because float doesn't have .dt.hour
        self.assertRaises(AttributeError, decompose_datetime, test_dataframe, "TIME", ["hour"])

    def test_absent_column(self):
        test_dataframe = pd.DataFrame({"TIME": [1, 2, 3], "OTHER": 1})
        self.assertRaises(KeyError, decompose_datetime, test_dataframe, "TEST", ["hour"])

    def test_recode_cyclical_datetime(self):
        # Test with the default datetime features instead of custom min/max
        time1 = '2019/03/11 00:00:00'
        time2 = '2019/03/11 18:00:00'
        freq = '6h'
        daterange = pd.date_range(time1, time2, freq=freq)
        test_dataframe = pd.DataFrame({'TIME': daterange, 'OTHER': 1})

        result = decompose_datetime(test_dataframe, components=['day', 'hour', 'minute'],
                                    cyclicals=['day', 'hour', 'minute'])

        expected = pd.DataFrame({
            'TIME': test_dataframe['TIME'],
            'OTHER': test_dataframe['OTHER'],
            'TIME_day_sin': np.sin(2 * np.pi * 11 / 31),  # always 11 out of 31
            'TIME_day_cos': np.cos(2 * np.pi * 11 / 31),  # always 11 out of 31
            'TIME_hour_sin': [0.0, 1.0, 0.0, -1.0],  # round trip around the clock
            'TIME_hour_cos': [1.0, 0.0, -1, 0.0],  # round trip around the clock
            'TIME_minute_sin': [0.0, 0.0, 0.0, 0.0],  # always 0
            'TIME_minute_cos': [1.0, 1.0, 1.0, 1.0],  # always 0

        })
        assert_frame_equal(result, expected)

    def test_min_higher_than_max(self):
        test_dataframe = pd.DataFrame({"TIME": [1, 2, 3], "OTHER": 1})
        self.assertRaises(ValueError, recode_cyclical_features, test_dataframe, ["TIME"],
                          cyclical_mins=[1], cyclical_maxes=[0])


if __name__ == '__main__':
    unittest.main()
