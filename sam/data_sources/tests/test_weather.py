import unittest
from sam.data_sources import read_knmi, read_openweathermap, read_regenradar
from sam import config
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
import pytest

try:
    owm_apikey = config['openweathermap']['apikey']
except KeyError:
    owm_apikey = None

try:
    rr_user = config['regenradar']['user']
    rr_password = config['regenradar']['password']
except KeyError:
    rr_user = None
    rr_password = None

skipowm = pytest.mark.skipif(owm_apikey is None,
                             reason="Openweathermap API key not found in .credentials file")
skiprr = pytest.mark.skipif(rr_user is None,
                            reason="Regenradar user+password not found in .credentials file")


class TestWeather(unittest.TestCase):

    # We don't test the actual content of the weather predictions, only the columns and dtypes
    # The TIME column is tested exactly, since it should be deterministic

    def test_read_knmi_hourly(self):
        result = read_knmi('2017-01-01 00:00:00', '2017-01-01 00:06:00', latitude=52.11,
                           longitude=5.18, freq='hourly', variables='default')
        self.assertEqual(result.columns.tolist(), ['RH', 'SQ', 'T', 'TIME'])

        expected_time = pd.Series(pd.date_range('2017-01-01', '2017-01-01 00:06:00', freq='H'))
        expected_time.name = 'TIME'
        assert_series_equal(expected_time, result['TIME'])

        expected_dtypes = [np.dtype('int64'), np.dtype('int64'), np.dtype('int64'),
                           np.dtype('<M8[ns]')]
        self.assertEqual(expected_dtypes, result.dtypes.tolist())

    def test_read_knmi_daily(self):
        result = read_knmi('2017-01-01 00:00:00', '2017-01-05 00:06:00', latitude=52.11,
                           longitude=5.18, freq='daily', variables=['RH', 'SQ'])
        self.assertEqual(result.columns.tolist(), ['RH', 'SQ', 'TIME'])
        expected_time = pd.Series(pd.date_range('2017-01-01', '2017-01-05', freq='D'))
        expected_time.name = 'TIME'
        assert_series_equal(expected_time, result['TIME'])

        expected_dtypes = [np.dtype('int64'), np.dtype('int64'), np.dtype('<M8[ns]')]
        self.assertEqual(expected_dtypes, result.dtypes.tolist())

    @skipowm
    def test_read_openweathermap(self):
        result = read_openweathermap(52.1, 6.2)
        expected_columns = ['cloud_coverage', 'pressure_groundlevel', 'humidity', 'pressure',
                            'pressure_sealevel', 'temp', 'temp_max', 'temp_min', 'rain_3h',
                            'wind_deg', 'wind_speed', 'TIME']
        expected_shape = (40, 12)

        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result.shape, expected_shape)

        # all observations have to be within now and 5 days from now.
        # It is possible for an observation to be before the present, b
        within_5_days = (result['TIME'] <= pd.Timestamp.now(tz='UTC') + pd.Timedelta('5 days')) & \
                        (result['TIME'] >= pd.Timestamp.now(tz='UTC'))
        self.assertTrue(within_5_days.all())

    @skiprr
    def test_read_regenradar(self):
        result = read_regenradar('2018-01-01', '2018-01-02 00:20:00', 52, 5.7, freq='5min')
        self.assertEqual(result.columns.tolist(), ['TIME', 'PRECIPITATION'])

        expected_time = pd.Series(pd.date_range('2018-01-01', '2018-01-02 00:20:00', freq='5min'))
        expected_time.name = 'TIME'
        assert_series_equal(expected_time, result['TIME'])

        expected_dtypes = [np.dtype('<M8[ns]'), np.dtype('float64')]
        self.assertEqual(result.dtypes.tolist(), expected_dtypes)

    def test_incorrect_knmi(self):
        self.assertRaises(Exception, read_knmi, '2017-01-01 00:00:00', '2017-01-01 00:06:00',
                          freq='monthly')

    @skipowm
    def test_incorrect_owm(self):
        self.assertRaises(Exception, read_openweathermap, {'lat': 52, 'lon': 5.5})
        self.assertRaises(Exception, read_openweathermap, None, 5.5)

    @skiprr
    def test_incorrect_rr(self):
        # Frequency too low
        self.assertRaises(Exception, read_regenradar, '2018-01-01', '2018-01-02 00:20:00',
                          freq='1min')
        # Incorrect dateformat
        self.assertRaises(Exception, read_regenradar, '20180101', '20180102')


if __name__ == '__main__':
    unittest.main()
