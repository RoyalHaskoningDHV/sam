import unittest
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from sam import config
from sam.data_sources import (
    read_knmi,
    read_knmi_stations,
    read_openweathermap,
    read_regenradar,
)

try:
    owm_apikey = config["openweathermap"]["apikey"]
except KeyError:
    owm_apikey = None

try:
    rr_user = config["regenradar"]["user"]
    rr_password = config["regenradar"]["password"]
except KeyError:
    rr_user = None
    rr_password = None

skipowm = pytest.mark.skipif(
    owm_apikey is None, reason="Openweathermap API key not found in .credentials file"
)
skiprr = pytest.mark.skipif(
    rr_user is None, reason="Regenradar user+password not found in .credentials file"
)


class TestWeather(unittest.TestCase):

    # We don't test the actual content of the weather predictions, only the columns and dtypes
    # The TIME column is tested exactly, since it should be deterministic
    def test_read_knmi_stations(self):
        result = read_knmi_stations()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            result.columns.tolist(),
            ["number", "longitude", "latitude", "altitude", "name"],
        )
        self.assertGreater(result.shape[0], 1)

    def test_read_knmi_hourly(self):
        result = read_knmi(
            "2016-03-07 06:00:00",
            "2016-03-07 12:00:00",
            latitude=52.11,
            longitude=5.18,
            freq="hourly",
            variables=["RH", "SQ", "N"],
        )
        self.assertEqual(result.columns.tolist(), ["RH", "SQ", "N", "TIME"])

        expected_time = pd.Series(
            pd.date_range("2016-03-07 06:00:00", "2016-03-07 12:00:00", freq="H")
        )
        expected_time.name = "TIME"
        assert_series_equal(expected_time, result["TIME"])

        expected_dtypes = [
            np.dtype("float64"),
            np.dtype("float64"),
            np.dtype("float64"),
            np.dtype("<M8[ns]"),
        ]
        self.assertEqual(expected_dtypes, result.dtypes.tolist())

    def test_read_knmi_daily(self):
        result = read_knmi(
            "2017-01-01 00:00:00",
            "2017-01-05 00:06:00",
            latitude=52.11,
            longitude=5.18,
            freq="daily",
            variables=["RH", "SQ"],
        )
        self.assertEqual(result.columns.tolist(), ["RH", "SQ", "TIME"])
        expected_time = pd.Series(pd.date_range("2017-01-01", "2017-01-05", freq="D"))
        expected_time.name = "TIME"
        assert_series_equal(expected_time, result["TIME"])

        expected_dtypes = [
            np.dtype("float64"),
            np.dtype("float64"),
            np.dtype("<M8[ns]"),
        ]
        self.assertEqual(expected_dtypes, result.dtypes.tolist())

    def test_nonan_knmi_station(self):

        # This station had all nan values for variable 'P' when I check it,
        # so I picked a relatively short and past moment. If knmi will fill
        # these values later on, the warning below will start to warn for this.
        # If this happens, we need to pick a new nan returning call.
        result = read_knmi(
            "2017-01-01 00:00:00",
            "2017-01-05 00:00:00",
            latitude=51.55,
            longitude=6.75,
            freq="hourly",
            variables=["P"],
            find_nonan_station=False,
        )

        if result.isna().sum().sum() == 0:
            warnings.warn(
                "The data already contains no nans, so "
                + "find_nonan_station unit test is worthless",
                UserWarning,
            )

        result = read_knmi(
            "2017-01-01 00:00:00",
            "2017-01-05 00:00:00",
            latitude=51.55,
            longitude=6.75,
            freq="hourly",
            variables=["P"],
            find_nonan_station=True,
        )
        self.assertEqual(result.isna().sum().sum(), 0)

    @skipowm
    def test_read_openweathermap(self):
        result = read_openweathermap(52.1, 6.2)
        expected_columns = [
            "cloud_coverage",
            "pressure_groundlevel",
            "humidity",
            "pressure",
            "pressure_sealevel",
            "temp",
            "temp_max",
            "temp_min",
            "rain_3h",
            "wind_deg",
            "wind_speed",
            "TIME",
        ]
        expected_shape = (40, 12)

        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result.shape, expected_shape)

        # all observations have to be within now and 5 days from now.
        # It is possible for an observation to be before the present, b
        within_5_days = (result["TIME"] <= pd.Timestamp.now(tz="UTC") + pd.Timedelta("5 days")) & (
            result["TIME"] >= pd.Timestamp.now(tz="UTC")
        )
        self.assertTrue(within_5_days.all())

    @skiprr
    def test_read_regenradar(self):
        result = read_regenradar("2018-01-01", "2018-01-02 00:20:00", 52, 5.7, freq="5min")
        self.assertEqual(result.columns.tolist(), ["TIME", "PRECIPITATION"])

        expected_time = pd.Series(pd.date_range("2018-01-01", "2018-01-02 00:20:00", freq="5min"))
        expected_time.name = "TIME"
        assert_series_equal(expected_time, result["TIME"])

        expected_dtypes = [np.dtype("<M8[ns]"), np.dtype("float64")]
        self.assertEqual(result.dtypes.tolist(), expected_dtypes)

    def test_incorrect_knmi(self):
        self.assertRaises(
            Exception,
            read_knmi,
            "2017-01-01 00:00:00",
            "2017-01-01 00:06:00",
            freq="monthly",
        )

    @skipowm
    def test_incorrect_owm(self):
        self.assertRaises(Exception, read_openweathermap, {"lat": 52, "lon": 5.5})
        self.assertRaises(Exception, read_openweathermap, None, 5.5)

    @skiprr
    def test_incorrect_rr(self):
        # Frequency too low
        self.assertRaises(
            Exception, read_regenradar, "2018-01-01", "2018-01-02 00:20:00", freq="1min"
        )
        # Incorrect dateformat
        self.assertRaises(Exception, read_regenradar, "20180101", "20180102")


if __name__ == "__main__":
    unittest.main()
