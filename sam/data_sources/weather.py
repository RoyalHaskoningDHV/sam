import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import datetime
import math
from sam.data_sources import knmy_stations
from sam.logging import log_dataframe_characteristics
from sam import config  # Credentials file
import logging
logger = logging.getLogger(__name__)


# Variables that are by default on a decimal scale: units of 0.1 instead of 1.0
knmy_decimal_variables = [
    'FH', 'FF', 'FX', 'T', 'T10N', 'TD', 'SQ', 'DR', 'RH', 'P',
    'FHVEC', 'FG', 'FHX', 'FHN', 'FHNH', 'TG', 'TN', 'TX', 'RHX',
    'PG', 'PX', 'PN', 'EV24'
]

# variables for which values < 0.05 are returned as -1
knmy_positive_variables = ['SQ', 'RH', 'RHX']


def _haversine(stations_row, lat2, lon2):
    """
    Helper function to calculate the distance between a station and a (lat, lon) position
    stations_row is a row of knmy_stations, which means it's a dataframe with shape (1, 3)
    `Credit for this solution goes to stackoverflow <https://stackoverflow.com/a/19412565>`_
    """
    lat1, lon1 = math.radians(stations_row['latitude']), math.radians(stations_row['longitude'])
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((lat2 - lat1) / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6373.0 * c  # radius of the earth, in km


def _try_parsing_date(text):
    """
    Helper function to try parsing text that either does or does not have a time
    To make the functions below easier, since often time is optional in the apis
    """
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found')


def read_knmi(start_date, end_date, latitude=52.11, longitude=5.18, freq='hourly',
              variables='default', find_nonan_station=False, preprocess=False):
    """
    Export historic variables from KNMI, either hourly or daily.
    There are many weather stations in the Netherlands, but this function will select the station
    that is physically closest to the desired location, and use that station.
    knmi only has historic data. Usually, the most recent datapoint is about half a day prior to
    the current time. If the `start_date` and/or `end_date` is after the most recent available
    datapoint, any datapoints that are not available will not be included in the results, not
    even as missing data.

    Parameters
    ----------
    start_date : str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d %H:%M:%S` or `%Y-%m-%d`
    end_date : str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d %H:%M:%S` or `%Y-%m-%d`
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
    find_nonan_station: bool, optional (defaut=False)
        by default (False), return the closest stations even if it includes nans.
        If True, return the closest station that does not include nans instead
    preprocess: boolm optional (default=False)
        by default (False), return variables in default units (often 0.1 mm).
        If true, data is scaled to whole units, and default values of -1 are mapped to 0

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
    stations = np.array(knmy_stations['number'][np.argsort(distances.values)])

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

    for si, station in enumerate(stations):
        # get data for this station
        _, _, _, knmi_raw = knmy.get_knmi_data(
            type=freq,
            stations=[station],
            start=start_date,
            end=end_date,
            inseason=False,
            variables=variables,
            parse=True)
        # save first station for later return if no no-nan station is found
        if si == 0:
            first_knmi_data = knmi_raw
        if (not find_nonan_station or
           (knmi_raw[variables].isna().sum().sum() == 0)):
            break

    # if there are no stations without nans, raise exception:
    if si == len(stations):
        knmi_raw = first_knmi_data
        raise RuntimeError(
            'Warning, no stations without nans found, ' +
            'returning requested station instead (including nans).')

    # log proximity of station
    logger.info('Got data from station %s' % station)
    if si > 0:
        logger.warning('Retrieved data is %d stations away ' % si +
                       'from the one requested. If this is too far away, ' +
                       'please run this function with return_nonan=False')

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

    if preprocess:
        for col in knmi.columns:
            if col in knmy_decimal_variables:
                knmi[col] = knmi[col].divide(10)
            if col in knmy_positive_variables:
                knmi[col] = knmi[col].clip(lower=0)

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
