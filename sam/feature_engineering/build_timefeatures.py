import pandas as pd


def build_timefeatures(start_time, end_time, freq, year=True, seasonal=True, weekly=True, daily=True):
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
    freq : str or DateOffset
        the frequency with which the time features are made
        See a list of frequency aliases here:
        https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
        frequencies can have multiples, e.g. "15 min" for 15 minutes
    year : boolean, optional (default=True)
        if this is true, then the "YEAR" feature will be created
    seasonal : boolean, optional (default=True)
        if this is true, then the "MONTH", "QUARTER" and "WEEK" features will be created
    weekly : boolean, optional (default=True)
        if this is true, then the "WEEKDAY" and "WEEKEND" (boolean) features will be created
    daily : boolean, optional (default=True)
        if this is true, then the "HOUR", "MINUTE" and "DAY_PERIOD" (string) features will be created

    Returns
    -------
    result : dataframe
        A dataframe with columns containing time features
        it always contains column TIME and the index will also be identical to TIME
        Furthermore, it contains columns such as YEAR, MONTH, etcetera, depending on the
        chosen inputs.
        All features are numeric other than TIME (which is datetime), and WEEKEND and DAY_PERIOD

    """
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

    return(result)
