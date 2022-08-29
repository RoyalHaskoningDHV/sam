import logging

import pandas as pd
from pandas import json_normalize
from sam import config  # Credentials file
from sam.logging_functions import log_dataframe_characteristics
from sam.data_sources.weather.utils import _try_parsing_date

logger = logging.getLogger(__name__)


def read_regenradar(
    start_date: str,
    end_date: str,
    latitude: float = 52.0237687,
    longitude: float = 5.5920412,
    freq: float = "5min",
    batch_size: str = "7D",
    crs: str = "EPSG:4326",
    **kwargs,
) -> pd.DataFrame:
    """
    Export historic precipitation from Nationale Regenradar.

    By default, this function collects the best-known information for a single point, given by
    latitude and longitude in coordinate system EPSG:4326 (WGS84). This can be configured
    using `**kwargs`, but this requires some knowledge of the underlying API.

    The parameters `agg=average`, `rasters=730d6675`, `srs=EPSG:4326m` are given to the API, as
    well as `start`, `end`, `window` given by `start_date`, `end_date`, `freq`. Lastly `geom`,
    which is `POINT+(latitude+longitude)`.
    Alternatively, a different geometry can be passed via the `geom` argument in `**kwargs`.
    A different coordinate system can be passed via the `srs` argument in `**kwargs`.
    This is a WKT string. For example: `geom='POINT+(191601+500127)', srs='epsg:28992'`.
    Exact information about the API specification and possible arguments is unfortunately unknown.

    Parameters
    ----------
    start_date: str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d` or `%Y-%m-%d %H:%M:%S`
    end_date: str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d` or `%Y-%m-%d %H:%M:%S`
    latitude: float, optional (default=52.11)
        latitude of the location from which to export weather. By default, use location of weather
        station De Bilt
    longitude: float, optional (default=5.18)
        longitude of the location from which to export weather. By default, use location of
        weather station De Bilt
    freq: str or DateOffset, default ‘5min’
        frequency of export. Minimum, and default frequency is every 5 minutes. To learn more
        about the frequency strings, see `this link
        <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__.
    batch_size: str, default '7D'
        batch size for collecting data from the API to avoid time-out. Default is 7 days.
    crs: str, default 'EPSG:4326'
        coordinate system for provided longitude (x) and latitude (y) values (or geometry by
        kwargs). Default is WGS84.
    kwargs: dict
        additional parameters passed in the url. Must be convertable to string. Any entries with a
        value of None will be ignored and not passed in the url.

    Returns
    -------
    result: dataframe
        Dataframe with column `PRECIPITATION` and column `TIME`.
        `PRECIPITATION` is the precipitation in the last 5 minutes, in mm.

    Examples
    --------
    >>> from sam.data_sources import read_regenradar
    >>> read_regenradar('2018-01-01', '2018-01-01 00:20:00')  # doctest: +SKIP
        TIME	PRECIPITATION
    0	2018-05-01 00:00:00	0.05
    1	2018-05-01 00:05:00	0.09
    2	2018-05-01 00:10:00	0.09
    3	2018-05-01 00:15:00	0.07
    4	2018-05-01 00:20:00	0.04

    >>> # Example of using alternative **kwargs
    >>> # For more info about these parameters, ask regenradar experts at RHDHV
    >>> read_regenradar(
    ...     '2018-01-01',
    ...     '2018-01-01 00:20:00',
    ...     boundary_type='MUNICIPALITY',
    ...     geom_id=95071,
    ...     geom=None,
    ... )  # doctest: +SKIP
        TIME	PRECIPITATION
    0	2018-05-01 00:00:00	0.00
    1	2018-05-01 00:05:00	0.00
    2	2018-05-01 00:10:00	0.00
    3	2018-05-01 00:15:00	0.00
    4	2018-05-01 00:20:00	0.00
    """
    import requests

    # convert to milliseconds, which the regenradar needs
    window = int(pd.tseries.frequencies.to_offset(freq).nanos / 1000000)
    if window < 300 * 1000:
        raise ValueError("The minimum window for read_regenradar is 300000")

    # will raise exception if the section does not appear in the config file
    user = config["regenradar"]["user"]
    password = config["regenradar"]["password"]

    logger.debug(
        f"Getting regenradar historic data: start_date={start_date}, "
        f"end_date={end_date}, latitude={latitude}, "
        f"longitude={longitude}, window={window}"
    )

    if isinstance(start_date, str):
        start_date = _try_parsing_date(start_date)
    if isinstance(end_date, str):
        end_date = _try_parsing_date(end_date)

    date_range = pd.date_range(start_date, end_date, freq=batch_size)

    result = []
    for date in date_range:
        start_date_, end_date_ = date, date + pd.Timedelta(batch_size)

        logging.debug(f"Getting regenradar data batch for {start_date_} to {end_date_}")

        regenradar_url = config["regenradar"]["url"]
        params = {
            "agg": "average",
            "rasters": "730d6675",
            "srs": crs,
            "start": str(start_date_),
            "stop": str(end_date_),
            "window": str(window),
            "geom": f"POINT+({longitude}+{latitude})",
        }
        params.update(kwargs)
        params = "&".join("%s=%s" % (k, v) for k, v in params.items() if v is not None)

        res = requests.get(regenradar_url + params, auth=(user, password))
        res = res.json()
        data = json_normalize(res, "data")

        # Time in miliseconds, convert to posixct
        data.columns = ["TIME", "PRECIPITATION"]
        data["TIME"] = pd.to_datetime(data["TIME"], unit="ms")

        result.append(data)
    data = pd.concat(result, ignore_index=True)

    # because of batch collection, there can be duplicate data or data outside of the date range
    data = (
        data.drop_duplicates()
        .loc[data["TIME"].between(start_date, end_date)]
        .reset_index(drop=True)
    )

    log_dataframe_characteristics(data, logging.DEBUG)
    return data
