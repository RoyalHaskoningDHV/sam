import pandas as pd
from pandas.io.json import json_normalize
import datetime
import math
from sam.logging import log_dataframe_characteristics
from sam import config  # Credentials file
import logging
logger = logging.getLogger(__name__)

"""
Location of all weather stations available in knmy, with coordinates
List of all weather stations can be found here:
http://projects.knmi.nl/datacentrum/catalogus/catalogus/content/nl-obs-surf-stationslijst.htm
"""
knmy_stations = pd.DataFrame({
    'number': [201, 203, 204, 205, 206, 207, 208, 209, 215, 211, 212, 225, 229, 235, 239, 240, 242,
               248, 249, 251, 252, 257, 258, 260, 267, 269, 270, 273, 275, 277, 278, 279, 280, 283,
               285, 286, 290, 308, 310, 311, 312, 313, 315, 316, 319, 320, 321, 323, 324, 330, 331,
               340, 343, 344, 348, 350, 356, 370, 375, 377, 380, 391],
    'latitude': [54.19, 52.22, 53.16, 55.25, 54.07, 53.37, 53.30, 52.28, 52.08, 53.49, 52.55,
                 52.28, 53.00, 52.55, 54.51, 52.18, 53.15, 52.38, 52.39, 53.23, 53.13, 52.30,
                 52.39, 52.06, 52.53, 52.27, 53.13, 52.42, 52.04, 53.25, 52.26, 52.44, 53.08,
                 52.04, 53.34, 53.12, 52.16, 51.23, 51.27, 51.23, 51.46, 51.30, 51.27, 51.39,
                 51.14, 51.56, 52.00, 51.32, 51.36, 51.59, 51.31, 51.27, 51.53, 51.57, 51.58,
                 51.34, 51.52, 51.27, 51.39, 51.12, 50.55, 51.30],
    'longitude': [2.56, 3.21, 3.38, 3.49, 4.01, 4.58, 5.57, 4.31, 4.26, 2.57, 3.49, 4.34, 4.45,
                  4.47, 4.42, 4.46, 4.55, 5.10, 4.59, 5.21, 3.13, 4.36, 5.24, 5.11, 5.23, 5.32,
                  5.46, 5.53, 5.53, 6.12, 6.16, 6.31, 6.35, 6.39, 6.24, 7.09, 6.54, 3.23, 3.36,
                  3.40, 3.37, 3.15, 4.00, 3.42, 3.50, 3.40, 3.17, 3.54, 4.00, 4.06, 4.08, 4.20,
                  4.19, 4.27, 4.56, 4.56, 5.09, 5.25, 5.42, 5.46, 5.47, 6.12]
})


def _haversine(stations_row, lat2, lon2):
    """ Helper function to calculate the distance between a station and a (lat, lon) position
    stations_row is a row of knmy_stations, which means it's a dataframe with shape (1, 3)
    Credit for this solution goes to https://stackoverflow.com/a/19412565 """
    lat1, lon1 = math.radians(stations_row['latitude']), math.radians(stations_row['longitude'])
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((lat2 - lat1) / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6373.0 * c  # radius of the earth, in km


def _try_parsing_date(text):
    """ Helper function to try parsing text that either does or does not have a time
    To make the functions below easier, since often time is optional in the apis"""
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found')


def read_knmi(start_date, end_date, latitude=52.11, longitude=5.18, freq='hourly',
              variables='default'):
    """Export historic variables from KNMI, either hourly or daily.
    There are many weather stations in the Netherlands, but this function will select the station
    that is physically closest to the desired location, and use that station.

    knmi only has historic data. Usually, the most recent datapoint is about half a day prior to
    the current time. If the start_date and/or end_date is after the most recent available
    datapoint, any datapoints that are not available will not be included in the results, not
    even as missing data.

    Parameters
    ----------
    start_date : str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format %Y-%m-%d %H:%M:%S or %Y-%m-%d %H:%M:%S
    end_date : str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format %Y-%m-%d %H:%M:%S or %Y-%m-%d %H:%M:%S
    latitude : float, optional (default=52.11)
        latitude of the location from which to export weather. By default, use location of weather
        station De Bilt
    longitude : float, optional (default=5.18)
        longitude of the location from which to export weather. By default, use location of
        weather station De Bilt
    freq: str, optional (default = 'hourly')
        frequency of export. Must be 'hourly' or 'daily'
    variables: list of str, optional (default='default')
        knmi-variables to export. See `all hourly variables here
        <https://projects.knmi.nl/klimatologie/uurgegevens/selectie.cgi>`_ or `all daily variables
        here <https://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi>`_
        by default, export [average temperature, sunshine duration, rainfall], which is
        ['RH', 'SQ', 'T'] for hourly, and ['RH', 'SQ', 'TG'] for daily

    Returns
    -------
    knmi: dataframe
        Dataframe with columns as in 'variables', and TIME column

    Examples
    --------
    >>> read_knmi('2018-01-01 00:00:00', '2018-01-01 06:00:00', 52.09, 5.09, 'hourly', ['SQ', 'T'])
        SQ	T	TIME
    0	0	87	2018-01-01 00:00:00
    1	0	85	2018-01-01 01:00:00
    2	0	71	2018-01-01 02:00:00
    3	0	78	2018-01-01 03:00:00
    4	0	80	2018-01-01 04:00:00
    5	0	75	2018-01-01 05:00:00
    6	0	69	2018-01-01 06:00:00
    """
    from knmy import knmy  # only needed for this function

    assert freq in ['hourly', 'daily']
    logger.debug(("Getting KNMI historic data: start_date={}, end_date={}, latitude={}, "
                  "longitude={}, freq={}, variables={}").
                 format(start_date, end_date, latitude, longitude, freq, variables))

    if variables == 'default' and freq == 'hourly':
        variables = ['RH', 'SQ', 'T']
    elif variables == 'default' and freq == 'daily':
        variables = ['RH', 'SQ', 'TG']

    # Provide 1 of 50 stations, find closest to specified coordinate
    distances = knmy_stations.apply(_haversine, axis=1, args=(latitude, longitude))
    station = knmy_stations['number'][distances.values.argmin()]

    if isinstance(start_date, str):
        start_date = _try_parsing_date(start_date)
    if isinstance(end_date, str):
        end_date = _try_parsing_date(end_date)

    if freq == 'hourly':
        # knmi API only works if the beginning hour is 0, and the ending hour is 23.
        # we still keep a backup so we can trim the unwanted results at the end.
        start_backup, end_backup = start_date, end_date
        start_date = start_date - datetime.timedelta(hours=1)
        start_date = start_date.replace(hour=0, minute=0)
        end_date = end_date.replace(hour=23, minute=0)

    _, _, _, knmi_raw = knmy.get_knmi_data(type=freq, stations=[station],
                                           start=start_date, end=end_date,
                                           inseason=False, variables=variables, parse=True)

    # First row are headers, so drop and fix types
    # Convert variables to float because pandas does not handle nans wit int
    dtype_dict = dict(zip(variables, [float] * len(variables)))
    knmi = knmi_raw.drop(0).reset_index(drop=True).astype(dtype_dict)

    if freq == 'hourly':
        knmi['HH'] = pd.to_numeric(knmi['HH'])  # needs to be numeric to subtract 1
        # Subtract 1 from HH since it runs from 1 to 24, which will make datetime conversion fail
        knmi['TIME'] = knmi['YYYYMMDD'].astype(str) + ' ' + (knmi['HH'] - 1).astype(str) + ":00:00"
    elif freq == 'daily':
        knmi['TIME'] = knmi['YYYYMMDD'].astype(str) + ' 00:00:00'

    knmi = knmi.drop(['STN', 'YYYYMMDD', 'HH'], axis=1, errors='ignore')
    knmi['TIME'] = pd.to_datetime(knmi['TIME'], format='%Y%m%d %H:%M:%S')

    if freq == 'hourly':
        # add the hour back that we subtracted a few lines earlier
        knmi['TIME'] = knmi['TIME'] + pd.Timedelta('1 hour')
        # Filter the unwanted results since we changed the start/end earlier
        knmi = knmi.loc[(knmi['TIME'] >= start_backup) & (knmi['TIME'] <= end_backup)]. \
            reset_index(drop=True)

    log_dataframe_characteristics(knmi, logging.DEBUG)
    return knmi


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
    >>> read_openweathermap(52.11, 5.18)
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

    logger.debug("Getting openweathermap forecast: latitude={}, longitude={}".
                 format(latitude, longitude))
    url = "https://api.openweathermap.org/data/2.5/forecast?units=metric&lat={}&lon={}&APPID={}". \
        format(latitude, longitude, apikey)
    res = requests.get(url).json()['list']
    data = json_normalize(res)

    data['TIME'] = pd.to_datetime(data['dt'], unit='s', utc=True)
    data = data.rename({
        'clouds.all': 'cloud_coverage',
        'main.grnd_level': 'pressure_groundlevel',
        'main.humidity': 'humidity',
        'main.pressure': 'pressure',
        'main.sea_level': 'pressure_sealevel',
        'main.temp': 'temp',
        'main.temp_max': 'temp_max',
        'main.temp_min': 'temp_min',
        'rain.3h': 'rain_3h',
        'wind.deg': 'wind_deg',
        'wind.speed': 'wind_speed'}, axis=1). \
        drop(['dt', 'dt_txt', 'weather', 'main.temp_kf', 'sys.pod'], axis=1)

    log_dataframe_characteristics(data, logging.DEBUG)
    return data


def read_regenradar(start_date, end_date, latitude=52.11, longitude=5.18, freq='5min', **kwargs):
    """
    Export historic precipitation from Nationale Regenradar.

    By default, this function collects the best-known information for a single point, given by
    latitude and longitude in coordinate system EPSG:4326 (WGS84). This can be configured
    using **kwargs, but this requires some knowledge of the underlying API.

    The parameters agg=average, rasters=730d6675, srs=EPSG:4326m are given to the API, as well as
    start, end, window given by start_date, end_date, freq. Lastly geom, which is
    `POINT+(latitude+longitude)`.
    Alternatively, a different geometry can be passed via the 'geom' argument in **kwargs.
    A different coordinate system can be passed via the 'srs' argument in **kwargs.
    This is a WKT string. For example: geom='POINT+(191601+500127)', srs='epsg:28992'.
    Exact information about the API specification and possible arguments is unfortunately unknown.

    Parameters
    ----------
    start_date: str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format %Y-%m-%d or %Y-%m-%d %H:%M:%S
    end_date: str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format %Y-%m-%d or %Y-%m-%d %H:%M:%S
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
    kwargs: dict
        additional parameters passed in the url. Must be convertable to string. Any entries with a
        value of None will be ignored and not passed in the url.

    Returns
    -------
    result: dataframe
        Dataframe with column PRECIPITATION and column TIME.
        PRECIPITATION is the precipitation in the last 5 minutes, in mm.

    Examples
    --------
    >>> from sam.data_sources import read_regenradar
    >>> read_regenradar('2018-01-01', '2018-01-01 00:20:00')
        TIME	PRECIPITATION
    0	2018-05-01 00:00:00	0.05
    1	2018-05-01 00:05:00	0.09
    2	2018-05-01 00:10:00	0.09
    3	2018-05-01 00:15:00	0.07
    4	2018-05-01 00:20:00	0.04

    >>> # Example of using alternative **kwargs
    >>> # For more info about these parameters, ask regenradar experts at RHDHV
    >>> read_regenradar('2018-01-01', '2018-01-01 00:20:00', boundary_type='MUNICIPALITY',
    >>>                 geom_id=95071, geom=None)
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
    assert window >= 300 * 1000, "The minimum window for read_regenradar is 300000"

    # will raise exception if the section does not appear in the config file
    user = config["regenradar"]["user"]
    password = config["regenradar"]["password"]

    logger.debug(("Getting regenradar historic data: start_date={}, end_date={}, latitude={}, "
                  "longitude={}, window={}").
                 format(start_date, end_date, latitude, longitude, window))
    if isinstance(start_date, str):
        start_date = _try_parsing_date(start_date)
    if isinstance(end_date, str):
        end_date = _try_parsing_date(end_date)

    regenradar_url = 'https://rhdhv.lizard.net/api/v3/raster-aggregates/?'
    params = {'agg': 'average',
              'rasters': '730d6675',
              'srs': 'EPSG:4326',
              'start': str(start_date),
              'stop': str(end_date),
              'window': str(window),
              'geom': 'POINT+({x}+{y})'.format(x=longitude, y=latitude)
              }
    params.update(kwargs)
    params = '&'.join('%s=%s' % (k, v) for k, v in params.items() if v is not None)

    res = requests.get(regenradar_url + params, auth=(user, password))
    res = res.json()
    data = json_normalize(res, 'data')

    # Time in miliseconds, convert to posixct
    data.columns = ['TIME', 'PRECIPITATION']
    data['TIME'] = pd.to_datetime(data['TIME'], unit='ms')
    log_dataframe_characteristics(data, logging.DEBUG)
    return data
