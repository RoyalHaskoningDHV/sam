import logging
from datetime import datetime
from typing import Callable, List, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_dataframe(df: pd.DataFrame):
    if df["ID"].isnull().sum() > 0:
        raise ValueError("ID column may not contain nans!")
    if df["TYPE"].isnull().sum() > 0:
        raise ValueError("TYPE column may not contain nans!")


def normalize_timestamps(
    df: pd.DataFrame,
    freq: str,
    start_time: Union[datetime, str] = "",
    end_time: Union[datetime, str] = "",
    round_method: str = "ceil",
    aggregate_method: Union[str, Callable, dict, List[Callable]] = "last",
    fillna_method: str = None,
):
    """
    Create a dataframe with all timestamps according to a given frequency. Fills in values
    for these timestamps from a given dataframe in SAM format.

    WARNING: This function makes assumptions about the data, and may not be safe for all datasets.
    For instance, a dataset with timestamps randomly distributed like a poisson process will
    significantly change when normalizing them, which may mean throwing away data.
    Furthermore, grouping measurements in the same timestamp may destroy cause-and-effect.
    For example, if a cause is measured at 15:58, and an effect is measured at 15:59, grouping
    them both at 16:00 makes it impossible to learn which came first.
    Use this function with caution, and generally mainly when the data already has normalized
    or close-to-normalized timestamps already.

    The process consists of four steps:

    Firstly, 'normalized' date ranges are created according to the required frequency.
    The start/end times of these date ranges can be given by start_time/end_time. If not given,
    the global minimum/maximum across all TYPE/ID is used. For example, if ID='foo' runs from 2017
    to 2019, and ID='bar' runs from 2018 to 2019, then ID='bar' will have missing values in
    the entirety of 2017.

    Secondly, all timestamps are rounded to the required frequency. For example, if the frequency
    is 1 hour, we may want the timestamp 19:45:12 to be rounded to 20:00:00.
    The method of rounding is ceiling by default, and is given by `round_method`.

    Thirdly, any timestamps with multiple measurements are aggregated.
    This is the last non-null value by default, and is given by `aggregate_method`.
    Other options are 'mean', 'median', 'first', and other pandas aggregation functions.

    Fourthly, any timestamps with missing values are filled. By default, no filling is done,
    and is given by `fillna_method`. The other options are backward filling and forward filling.
    ('bfill' and 'ffill')

    Parameters
    ----------
    df: pandas dataframe with TIME, TYPE, ID and VALUE columns, shape = (nrows, 4)
        Dataframe from which the values are created

    freq: str or DateOffset
        The required frequency for the time features.
        frequencies can have multiples, e.g. "15 min" for 15 minutes
        `See here for options
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`_

    start_time: str or datetime-like, optional (default = '')
        the start time of the period to create features over
        if string, the format `%Y-%m-%d %H:%M:%S` will always work
        Pandas also accepts other formats, or a `datetime` object

    end_time: str or datetime-like, optional (default = '')
        the end time of the period to create features over
        if string, the format `%Y-%m-%d %H:%M:%S` will always work
        Pandas also accepts other formats, or a `datetime` object

    round_method: string, optional (default = 'floor')
        How to group the times in bins. By default, rows are grouped by
        flooring them to the frequency (e.g.: if frequency is hourly, the timestamp 18:59 will
        be grouped together with 18:01, and the TIME will be set to 18:00)
        The options are:

        - 'floor': Group times by flooring to the nearest frequency
        - 'ceil': Group times by ceiling to the nearest frequency
        - 'round': Group times by rounding to the nearest frequency

        Ceiling is the option that is the safest to prevent leakage: It will guarantee that a
        value in the output will have a `TIME` that is not before the time that it
        actually occurred.

    aggregate_method: function, string, dictionary, list of string/functions (default = 'last')
        Method that is used to aggregate values when multiple values fall
        within a specified frequency region.
        For example, when you have data per 5 minutes, but you're creating a
        an hourly frequency, the values need to be aggregated.
        Can be strings such as mean, sum, min, max, or a function.
        `See also
        <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.aggregate.html>`_

    fillna_method: string, optional (default = None)
        Method used to fill NA values, must be an option from `pd.DataFrame.fillna`.
        Options are: 'backfill', 'bfill', 'pad', 'ffill', None
        `See also
        <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html>`_

    Returns
    -------
    complete_df: pandas dataframe,
        shape `(length(TIME) * length(unique IDs) * length(unique TYPEs, 4))`
        dataframe containing all possible combinations of timestamps and IDs and TYPEs
        with selected frequency, aggregate method and fillna method

    Examples
    --------
    >>> from sam.preprocessing import normalize_timestamps
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
    ...                             datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 46)],
    ...                    'ID': "SENSOR",
    ...                    'TYPE': "DEPTH",
    ...                    'VALUE': [1, 20, 3, 20]})
    >>>
    >>> normalize_timestamps(df, freq = "15 min", end_time="2018-06-09 12:15:00")
                     TIME      ID   TYPE  VALUE
    0 2018-06-09 11:15:00  SENSOR  DEPTH    1.0
    1 2018-06-09 11:30:00  SENSOR  DEPTH    NaN
    2 2018-06-09 11:45:00  SENSOR  DEPTH    3.0
    3 2018-06-09 12:00:00  SENSOR  DEPTH   20.0
    4 2018-06-09 12:15:00  SENSOR  DEPTH    NaN

    >>> from sam.preprocessing import normalize_timestamps
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
    ...                             datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 46)],
    ...                    'ID': "SENSOR",
    ...                    'TYPE': "DEPTH",
    ...                    'VALUE': [1, 20, 3, 20]})
    >>>
    >>> normalize_timestamps(df, freq = "15 min", end_time="2018-06-09 12:15:00",
    ...                     aggregate_method = "mean", fillna_method="ffill")
                     TIME      ID   TYPE  VALUE
    0 2018-06-09 11:15:00  SENSOR  DEPTH    1.0
    1 2018-06-09 11:30:00  SENSOR  DEPTH    1.0
    2 2018-06-09 11:45:00  SENSOR  DEPTH   11.5
    3 2018-06-09 12:00:00  SENSOR  DEPTH   20.0
    4 2018-06-09 12:15:00  SENSOR  DEPTH   20.0
    """

    if df.empty:
        raise ValueError("No dataframe found")
    df = df.copy()

    _validate_dataframe(df)

    original_rows = df.shape[0]
    original_nas = df["VALUE"].isna().sum()
    timezone = df["TIME"].dt.tz

    fillna_options = ["backfill", "bfill", "pad", "ffill", None]
    if fillna_method not in fillna_options:
        raise ValueError("fillna_method not in {}".format(str(fillna_options)))

    round_method_options = ["floor", "round", "ceil"]
    if round_method == "floor":
        df["TIME"] = df["TIME"].dt.floor(freq)  # Technically, this has no effect on the result
    elif round_method == "round":
        df["TIME"] = df["TIME"].dt.round(freq)
    elif round_method in "ceil":
        df["TIME"] = df["TIME"].dt.ceil(freq)
    else:
        raise ValueError("round_method not in {}".format(str(round_method_options)))

    if not start_time:
        start_time = df["TIME"].min()
    if not end_time:
        end_time = df["TIME"].max()

    logger.debug(
        "Normalizing timestamps: freq={}, start_time={}, end_time={}, "
        "aggregate_method={}, fillna_method={}, round_times={}".format(
            freq, start_time, end_time, aggregate_method, fillna_method, round_method
        )
    )

    ids, types, time = pd.core.reshape.util.cartesian_product(
        [
            df["ID"].unique(),
            df["TYPE"].unique(),
            pd.date_range(start=start_time, end=end_time, freq=freq, tz=timezone),
        ]
    )

    complete_df = pd.DataFrame(dict(TIME=time, ID=ids, TYPE=types), columns=["TIME", "ID", "TYPE"])
    complete_df["TIME"] = complete_df["TIME"].dt.floor(freq)

    # Function currently groups based on first left matching frequency,
    # can be set to the first right frequency within the Grouper function
    df = df.groupby([pd.Grouper(key="TIME", freq=freq), "ID", "TYPE"]).agg(
        {"VALUE": aggregate_method}
    )

    df = df.reset_index(drop=False)

    complete_df = complete_df.merge(df, how="left", on=["TIME", "ID", "TYPE"])

    logger.debug("Number of missings before fillna: {}".format(complete_df["VALUE"].isna().sum()))

    if fillna_method:
        complete_df["VALUE"] = complete_df.groupby(["ID", "TYPE"])["VALUE"].fillna(
            method=fillna_method
        )

    logger.info(
        "Dataframe changed because of normalize_timestamps: "
        "Previously it had {} rows, now it has {}".format(original_rows, complete_df.shape[0])
    )
    logger.info(
        "Also, the VALUE column previously had {} missing values, now it has {}".format(
            original_nas, complete_df["VALUE"].isna().sum()
        )
    )

    return complete_df
