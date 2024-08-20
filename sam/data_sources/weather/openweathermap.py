import logging

import pandas as pd
from pandas import json_normalize
from sam import config  # Credentials file
from sam.logging_functions import log_dataframe_characteristics

logger = logging.getLogger(__name__)


def read_openweathermap(latitude=52.11, longitude=5.18):
    """
    Use openweathermap API to obtain a weather forecast. This forecast has a frequency of 3 hours,
    with a total of 39 observations, meaning the forecast is up to 5 days in the future.
    The resulting timestamp always uses UTC.

    Parameters
    ----------
    latitude: float, optional (default=52.11)
        latitude of the location from which to export weather. By default, use location of weather
        station De Bilt
    longitude: float, optional (default=5.18)
        longitude of the location from which to export weather. By default, use location of
        weather station De Bilt

    Returns
    -------
    forecast: dataframe with TIME column, containing the time of that specific forecast,
        with timezone UTC. And the following columns:

        - cloud_coverage, in %
        - humidity, in %
        - pressure: generally same as pressure_sealevel, in hPa
        - pressure_groundlevel, in hPa
        - pressure_sealevel, in hPa
        - temp, in celcius
        - temp_max, in celcius
        - temp_min, in celcius
        - rain_3h: volume of the last 3h, in mm
        - wind_deg: wind direction in degrees (meteorological)
        - wind_speed, in meter/sec

    Examples
    --------
    >>> read_openweathermap(52.11, 5.18)  # doctest: +SKIP
        cloud_coverage	pressure_groundlevel	humidity	pressure	pressure_sealevel	temp	temp_max	\
            temp_min	rain_3h	wind_deg	wind_speed	TIME
    0	92	991.91	95	992.77	992.77	8.82	8.82	7.20	1.005	225.510	11.82	2019-03-07 15:00:00
    1	92	991.57	91	992.55	992.55	8.01	8.01	6.79	0.280	223.501	13.01	2019-03-07 18:00:00
    ...
    39	80	1009.42	73	1010.39	1010.39	8.41	8.41	8.41	0.090	204.502	10.28	2019-03-12 12:00:00
    """
    import requests

    # Will raise exception if this section does not appear in the config file
    apikey = config["openweathermap"]["apikey"]

    logger.debug(
        "Getting openweathermap forecast: latitude={}, longitude={}".format(latitude, longitude)
    )
    url = (
        "https://api.openweathermap.org/data/2.5/"
        f"forecast?units=metric&lat={latitude}&lon={longitude}&APPID={apikey}"
    )
    res = requests.get(url).json()["list"]
    data = json_normalize(res)

    data["TIME"] = pd.to_datetime(data["dt"], unit="s", utc=True)
    data = data.rename(
        {
            "clouds.all": "cloud_coverage",
            "main.grnd_level": "pressure_groundlevel",
            "main.humidity": "humidity",
            "main.pressure": "pressure",
            "main.sea_level": "pressure_sealevel",
            "main.temp": "temp",
            "main.temp_max": "temp_max",
            "main.temp_min": "temp_min",
            "rain.3h": "rain_3h",
            "wind.deg": "wind_deg",
            "wind.speed": "wind_speed",
        },
        axis=1,
    ).drop(["dt", "dt_txt", "weather", "main.temp_kf", "sys.pod"], axis=1)

    log_dataframe_characteristics(data, logging.DEBUG)
    return data
