import datetime
import logging
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests as req
from sam.data_sources.weather.utils import _haversine, _try_parsing_date
from sam.logging_functions import log_dataframe_characteristics

logger = logging.getLogger(__name__)

urls = {
    "daily": "https://www.daggegevens.knmi.nl/klimatologie/daggegevens",
    "hourly": "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens",
    "daily_rain": "https://www.daggegevens.knmi.nl/klimatologie/monv/reeksen",  # not implemented
}

# Variables that are by default on a decimal scale: units of 0.1 instead of 1.0
knmi_decimal_variables = [
    "FH",
    "FF",
    "FX",
    "T",
    "T10N",
    "TD",
    "SQ",
    "DR",
    "RH",
    "P",
    "FHVEC",
    "FG",
    "FHX",
    "FHN",
    "FHNH",
    "TG",
    "TN",
    "TX",
    "RHX",
    "PG",
    "PX",
    "PN",
    "EV24",
]

# variables for which values < 0.05 are returned as -1
knmi_positive_variables = ["SQ", "RH", "RHX"]


def _prepare_input(input):
    """Function to prepare KNMI API parameters
    Format of parameters should be for example 'RH:TX',
    where we prefer the ['RH', 'TX'] syntax.
    If parameter is None, the default 'ALL' is used
    """
    if input is None:
        input = "ALL"
    elif isinstance(input, int):
        input = str(input)  # station numbers
    elif isinstance(input, list):
        input = ":".join([str(x) for x in input])
    return input


def _parse_knmi_measurements(knmi_raw, freq, start=None, end=None):
    """Parse raw knmi data to a dataframe

    Parameters
    ----------
    knmi_raw : str
        Raw data from the KNMI API (text)
    freq : str
        Frequency of the data, either 'hourly' or 'daily'
    start : datetime.datetime, optional (default=None)
        Start date of the data
        Only used if freq='hourly'
    end : datetime.datetime, optional (default=None)
        End date of the data
        Only used if freq='hourly'

    Returns
    -------
    knmi : pandas.DataFrame
        Dataframe with KNMI data
    """
    # Parse raw data
    lines = knmi_raw.splitlines()
    header = [li.strip("#").replace(" ", "") for li in lines if "#" in li]
    columns = header[-1].split(",")

    knmi = pd.read_csv(StringIO(knmi_raw.replace(" ", "")), skiprows=len(header) - 1, header=0)
    knmi.columns = columns

    if freq == "hourly":
        knmi["HH"] = pd.to_numeric(knmi["HH"])  # needs to be numeric to subtract 1
        # Subtract 1 from H since it runs from 1 to 24, which will make datetime conversion fail
        knmi["TIME"] = knmi["YYYYMMDD"].astype(str) + " " + (knmi["HH"] - 1).astype(str) + ":00:00"
    elif freq == "daily":
        knmi["TIME"] = knmi["YYYYMMDD"].astype(str) + " 00:00:00"
    else:
        raise ValueError('freq must be either "hourly" or "daily"')

    knmi = knmi.drop(["YYYYMMDD", "HH"], axis=1, errors="ignore")
    knmi["TIME"] = pd.to_datetime(knmi["TIME"], format="%Y%m%d %H:%M:%S")

    if freq == "hourly":
        # add the hour back that we subtracted a few lines earlier
        knmi["TIME"] = knmi["TIME"] + pd.Timedelta("1 hour")
        # Filter the unwanted results since we changed the start/end earlier
        knmi = knmi.loc[(knmi["TIME"] >= start) & (knmi["TIME"] <= end)].reset_index(drop=True)
    return knmi


def _parse_knmi_stations(knmi_raw):
    """Parse the station data from raw data text"""
    lines = knmi_raw.splitlines()
    # for each line, remove space parts of length greater that 1
    lines = [line.strip("# ") for line in lines]
    lines = [re.sub(r"\s{2,}", ";", line) for line in lines]
    # find index of first line that starts with STN
    first_line_index = [i for i, line in enumerate(lines) if line.startswith("STN")][0]
    lines_after_header = lines[first_line_index:]
    # all lines are relevant up to those that contain ':'
    n_lines = [i for i, line in enumerate(lines_after_header) if ":" in line][0]
    lines = lines[(1 + first_line_index) : (n_lines + first_line_index)]

    # transform to dataframe
    df = pd.read_csv(StringIO("\n".join(lines)), skiprows=0, header=None, sep=";")
    df.columns = ["number", "longitude", "latitude", "altitude", "name"]
    return df


def _preprocess_knmi(knmi):
    """Preprocessing function for KNMI data
    Transforms data to a conventional scale
    """
    knmi_prep = knmi.copy()
    for col in knmi_prep.columns:
        if col in knmi_decimal_variables:
            knmi_prep[col] = knmi_prep[col].divide(10)
        if col in knmi_positive_variables:
            knmi_prep[col] = knmi_prep[col].clip(lower=0)
    return knmi_prep


def read_knmi_station_data(
    start_date="2021-01-01",
    end_date="2021-01-02",
    stations=None,
    freq="daily",
    variables="default",
    parse=True,
    preprocess=True,
):
    """Read KNMI data for specific station
    To find station numbers, use `sam.data_sources.read_knmi_stations`,
    or use `sam.data_sources.read_knmi` to use lat/lon and find closest station

    Source:
    https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

    Parameters
    ----------
    start_date : str or datetime-like
        the start time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d %H:%M:%S` or `%Y-%m-%d`
    end_date : str or datetime-like
        the end time of the period from which to export weather
        if str, must be in the format `%Y-%m-%d %H:%M:%S` or `%Y-%m-%d`
    stations : int, string, list or None
        station number or list of station numbers, either int or string
        if None, data from all stations is returned
    freq : str, optional (default = 'daily')
        frequency of export. Must be 'hourly' or 'daily'
    variables : str, None or list, optional (default='default')
        knmi-variables to export. See `all hourly variables here
        <https://www.daggegevens.knmi.nl/klimatologie/uurgegevens>`_ or `all daily variables
        here <https://www.daggegevens.knmi.nl/klimatologie/daggegevens>`_
        by default, export [average temperature, sunshine duration, rainfall], which is
        ['RH', 'SQ', 'T'] for hourly, and ['RH', 'SQ', 'TG'] for daily
        If None, all variables will be collected
    preprocess : bool, optional (default=False)
        by default (False), return variables in default units (often 0.1 mm).
        If true, data is scaled to whole units, and default values of -1 are mapped to 0
    parse : bool, optional (default=True)
        if True, parse the data to a pandas dataframe
        Only use False for debugging purposes
    Returns
    -------
    knmi: dataframe
        Dataframe with columns as in 'variables', and STN, TIME columns

    """

    if variables == "default" and freq == "hourly":
        variables = ["RH", "SQ", "T"]
    elif variables == "default" and freq == "daily":
        variables = ["RH", "SQ", "TG"]

    url = urls[freq]
    stns = _prepare_input(stations)
    vars = _prepare_input(variables)

    if isinstance(start_date, str):
        start_date = _try_parsing_date(start_date)
    if isinstance(end_date, str):
        end_date = _try_parsing_date(end_date)

    if freq == "hourly":
        # knmi API only works if the beginning hour is 0, and the ending hour is 23.
        # we still keep a backup so we can trim the unwanted results at the end.
        start_backup, end_backup = start_date, end_date
        start_date = start_date - datetime.timedelta(hours=1)
        start_date = start_date.replace(hour=0, minute=0)
        end_date = end_date.replace(hour=23, minute=0)
    else:
        start_backup, end_backup = start_date, end_date

    params = dict(start=start_date, end=end_date, vars=vars, stns=stns, fmt="csv")

    # get data
    with req.Session() as s:
        response = s.get(url, params=params)
    # get text from response if executed correctly
    if response.status_code == 200:
        knmi_raw = response.text
    else:
        raise ValueError("Request failed with status code {}".format(response.status_code))

    # Parse raw data
    if parse:
        knmi = _parse_knmi_measurements(knmi_raw, freq, start=start_backup, end=end_backup)
    else:
        return knmi_raw

    # (Optional) preprocessing
    numvar = [var for var in knmi.columns if var not in ["STN", "TIME"]]
    for var in numvar:
        knmi[var] = knmi[var].astype(float)
    if preprocess:
        knmi = _preprocess_knmi(knmi)

    return knmi


def read_knmi_stations():
    """Function to get all KNMI stations from API"""
    # empty request, containing stations data
    df = read_knmi_station_data(
        stations=["ALL"], start_date="2021-01-02", end_date="2021-01-01", parse=False
    )
    return _parse_knmi_stations(df)


def read_knmi(
    start_date,
    end_date,
    latitude=52.11,
    longitude=5.18,
    freq="hourly",
    variables="default",
    find_nonan_station=False,
    preprocess=False,
    drop_station=True,
):
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
    variables: str, None or list, optional (default='default')
        knmi-variables to export. See `all hourly variables here
        <https://www.daggegevens.knmi.nl/klimatologie/uurgegevens>`_ or `all daily variables
        here <https://www.daggegevens.knmi.nl/klimatologie/daggegevens>`_
        by default, export [average temperature, sunshine duration, rainfall], which is
        ['RH', 'SQ', 'T'] for hourly, and ['RH', 'SQ', 'TG'] for daily
        If None, all variables will be collected
    find_nonan_station : bool, optional (defaut=False)
        by default (False), return the closest stations even if it includes nans.
        If True, return the closest station that does not include nans instead
    preprocess : bool, optional (default=False)
        by default (False), return variables in default units (often 0.1 mm).
        If true, data is scaled to whole units, and default values of -1 are mapped to 0
    drop_station : bool, optional (default=True)
        by default (True), drop 'STN' column from result.
        If False, the returned dataframe will contain a column STN with station number
        This station number will be the same for all rows since this function returns only
        data for the closest station. To get data of multiple stations,
        try `read_knmi_station_data`

    Returns
    -------
    knmi: dataframe
        Dataframe with columns as in 'variables', and TIME column

    Examples
    --------
    >>> read_knmi('2018-01-01 00:00:00', '2018-01-01 06:00:00', 52.09, 5.09, 'hourly', ['SQ', 'T'])
        SQ     T                TIME
    0  0.0  87.0 2018-01-01 00:00:00
    1  0.0  85.0 2018-01-01 01:00:00
    2  0.0  71.0 2018-01-01 02:00:00
    3  0.0  78.0 2018-01-01 03:00:00
    4  0.0  80.0 2018-01-01 04:00:00
    5  0.0  75.0 2018-01-01 05:00:00
    6  0.0  69.0 2018-01-01 06:00:00
    """

    if freq not in ["hourly", "daily"]:
        raise ValueError("freq must be 'hourly' or 'daily'")
    logger.debug(
        (
            "Getting KNMI historic data: start_date={}, end_date={}, latitude={}, "
            "longitude={}, freq={}, variables={}"
        ).format(start_date, end_date, latitude, longitude, freq, variables)
    )

    # Find closest station to specified coordinate
    knmi_stations = read_knmi_stations()
    distances = knmi_stations.apply(_haversine, axis=1, args=(latitude, longitude))
    stations = np.array(knmi_stations["number"][np.argsort(distances.values)])

    for si, station in enumerate(stations):
        # get data for this station
        knmi = read_knmi_station_data(
            freq=freq,
            stations=[station],
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            preprocess=preprocess,
        )
        # save first station for later return if no no-nan station is found
        if si == 0:
            first_knmi_data = knmi
        if not find_nonan_station or (knmi[variables].isna().sum().sum() == 0):
            break

    # if there are no stations without nans, raise exception:
    if si == len(stations):
        knmi = first_knmi_data
        raise RuntimeError(
            "Warning, no stations without nans found, "
            + "returning requested station instead (including nans)."
        )

    # drop station columns (for backward compatability)
    if drop_station:
        knmi = knmi.drop("STN", axis=1)

    # log proximity of station
    logger.info("Got data from station %s" % station)
    if si > 0:
        logger.warning(
            "Retrieved data is %d stations away " % si
            + "from the one requested. If this is too far away, "
            + "please run this function with return_nonan=False"
        )

    log_dataframe_characteristics(knmi, logging.DEBUG)
    return knmi
