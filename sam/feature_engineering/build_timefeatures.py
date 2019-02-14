import pandas as pd
from sam.logging import log_dataframe_characteristics
import logging
logger = logging.getLogger(__name__)


def build_timefeatures(start_time, end_time, freq=None, year=True, seasonal=True, weekly=True,
                       daily=True):
    """
    Given a start time, end time, and frequency, in pandas format,
    create several time features, such as month, year, weekday, etcetera

    Parameters
    ----------
    start_time : str or datetime-like
        the start time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object
    end_time : str or datetime-like
        the end time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object
    freq : str or DateOffset (default=None)
        the frequency with which the time features are made
        See a list of frequency aliases here:
        https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        frequencies can have multiples, e.g. "15 min" for 15 minutes
        The default defaults to "D", which is one day
    year : boolean, optional (default=True)
        if this is true, then the "YEAR" feature will be created
    seasonal : boolean, optional (default=True)
        if this is true, then the "MONTH", "QUARTER" and "WEEK" features will be created
    weekly : boolean, optional (default=True)
        if this is true, then the "WEEKDAY" and "WEEKEND" (boolean) features will be created
    daily : boolean, optional (default=True)
        if this is true, then the "HOUR", "MINUTE" and "DAY_PERIOD" (string) features will
        be created

    Returns
    -------
    result : dataframe
        A dataframe with columns containing time features
        it always contains column TIME and the index will also be identical to TIME
        Furthermore, it contains columns such as YEAR, MONTH, etcetera, depending on the
        chosen inputs.
        All features are numeric other than TIME (which is datetime), and WEEKEND and DAY_PERIOD

    Examples
    --------
    >>> from sam.feature_engineering import build_timefeatures
    >>> build_timefeatures("28-12-2018", "01-01-2019", freq="11 H").reset_index(drop=True)
        TIME                YEAR    MONTH   QUARTER WEEK    WEEKDAY WEEKEND HOUR MINUTE  DAY_PERIOD
    0   2018-12-28 00:00:00 2018    12      4       52      4       False   0    0       night
    1   2018-12-28 11:00:00 2018    12      4       52      4       False   11   0       afternoon
    2   2018-12-28 22:00:00 2018    12      4       52      4       False   22   0       night
    3   2018-12-29 09:00:00 2018    12      4       52      5       True    9    0       morning
    4   2018-12-29 20:00:00 2018    12      4       52      5       True    20   0       evening
    5   2018-12-30 07:00:00 2018    12      4       52      6       True    7    0       morning
    6   2018-12-30 18:00:00 2018    12      4       52      6       True    18   0       evening
    7   2018-12-31 05:00:00 2018    12      4       1       0       False   5    0       night
    8   2018-12-31 16:00:00 2018    12      4       1       0       False   16   0       afternoon
    """
    logger.debug("Creating timefeatures from {} to {} with freq {}".
                 format(start_time, end_time, freq))
    logger.debug("Timefeatures will be year: {}, seasonal: {}, weekly: {}, daily: {}".
                 format(year, seasonal, weekly, daily))

    times = pd.date_range(start_time, end_time, freq=freq)
    assert times.size > 0

    result = pd.DataFrame({"TIME": times}).set_index(times)

    if year:
        result = result.assign(YEAR=result['TIME'].dt.year)

    if seasonal:
        result = result.assign(MONTH=result['TIME'].dt.month)
        result = result.assign(QUARTER=result['TIME'].dt.quarter)
        result = result.assign(WEEK=result['TIME'].dt.weekofyear)

    if weekly:
        result = result.assign(WEEKDAY=result['TIME'].dt.weekday)
        result = result.assign(WEEKEND=result['TIME'].dt.weekday > 4)

    if daily:
        day_period = {
            # https://stackoverflow.com/a/45928598
            **dict.fromkeys([7, 8, 9, 10], "morning"),
            **dict.fromkeys([11, 12, 13, 14, 15, 16], "afternoon"),
            **dict.fromkeys([17, 18, 19, 20], "evening"),
            **dict.fromkeys([21, 22, 23, 0, 1, 2, 3, 4, 5, 6], "night")
        }

        result = result.assign(HOUR=result['TIME'].dt.hour)
        result = result.assign(MINUTE=result['TIME'].dt.minute)
        result = result.assign(DAY_PERIOD=result['HOUR'].map(day_period))

    logger.info("Created timefeatures:")
    log_dataframe_characteristics(result)

    return(result)
