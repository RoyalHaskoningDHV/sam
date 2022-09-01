import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_dst(timestamps_series: pd.Series):
    """
    Find possible conflicts due to daylight savings time, by
    labeling timestamps_series. This converts a series of timestamps to
    a series of strings. The strings are either 'normal',
    'to_summertime', or 'to_wintertime'.
    to_summertime happens the last sunday morning of march,
    from 2:00 to 2:59.
    to_wintertime happens the last sunday morninig of october,
    from 2:00 to 2:59.
    These can be possible problems because they either happen 2
    or 0 times. to_summertime should therefore be impossible.

    Parameters
    ----------
    timestamps_series: pd.Series, shape = (n_inputs,)
        a series of pandas timestamps

    Returns
    -------
    labels: string, array-like, shape = (n_inputs,)
        a numpy array of strings, that are all either
        'normal', 'to_summertime', or 'to_wintertime'

    Examples
    --------
    >>> from sam.preprocessing import label_dst
    >>> import pandas as pd
    >>>
    >>> daterange = pd.date_range('2019-10-27 01:00:00', '2019-10-27 03:00:00', freq='15min')
    >>> date_labels = label_dst(pd.Series(daterange))
    >>>
    >>> pd.DataFrame({'TIME' : daterange,
    ...               'LABEL': date_labels})
                     TIME          LABEL
    0 2019-10-27 01:00:00         normal
    1 2019-10-27 01:15:00         normal
    2 2019-10-27 01:30:00         normal
    3 2019-10-27 01:45:00         normal
    4 2019-10-27 02:00:00  to_wintertime
    5 2019-10-27 02:15:00  to_wintertime
    6 2019-10-27 02:30:00  to_wintertime
    7 2019-10-27 02:45:00  to_wintertime
    8 2019-10-27 03:00:00         normal
    """
    logger.debug("Labeling dst on {} timestamps".format(timestamps_series.size))
    last_sunday_morning = (
        (timestamps_series.dt.day >= 25)
        & (timestamps_series.dt.weekday == 6)
        & (timestamps_series.dt.hour == 2)
    )
    result = np.where(
        (last_sunday_morning) & (timestamps_series.dt.month == 3),
        "to_summertime",
        np.where(
            (last_sunday_morning) & (timestamps_series.dt.month == 10),
            "to_wintertime",
            "normal",
        ),
    )

    valuecounts = str(pd.Series(result).value_counts()).replace("\n", ", ")
    logger.debug("labeldst output values: {}".format(valuecounts))
    return result


def average_winter_time(data: pd.DataFrame, tmpcol: str = "tmp_UNID"):
    """
    Solve duplicate timestamps in wintertime, by averaging them
    Because the to_wintertime hour happens twice, there can be duplicate timestamps
    This function removes those duplicates by averaging the VALUE column
    All other columns are used as group-by columns

    Parameters
    ----------
    data: pandas Dataframe
        must have columns TIME, VALUE, and optionally others like ID and TYPE.
    tmpcol: string, optional (default='tmp_UNID')
        temporary columnname that is created in dataframe. This columnname cannot
        exist in the dataframe already

    Returns
    -------
    data: pandas Dataframe
        The same dataframe as was given in input, but with duplicate timestamps
        removed, if they happened during the wintertime duplicate hour

    Examples
    --------
    >>> from sam.preprocessing import average_winter_time
    >>> import numpy as np
    >>>
    >>> daterange = pd.date_range('2019-10-27 01:45:00', '2019-10-27 03:00:00', freq='15min')
    >>> test_df = pd.DataFrame({"TIME": daterange.values[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]],
    ...                         "VALUE": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])})
    >>> average_winter_time(test_df)
                     TIME  VALUE
    0 2019-10-27 01:45:00    0.0
    1 2019-10-27 02:00:00    1.5
    2 2019-10-27 02:15:00    3.5
    3 2019-10-27 02:30:00    5.5
    4 2019-10-27 02:45:00    7.5
    5 2019-10-27 03:00:00    9.0
    """
    warnings.warn(
        "DEPRECATED: Convert to UTC using pandas Series.dt.tz_convert() and use the"
        " `timezone`-parameter of `decompose_datetime()` to extract local time features if needed",
        DeprecationWarning,
    )

    if tmpcol in data.columns:
        raise ValueError("tmpcol already exists in dataframe")

    logging.debug("Now averaging wintertime")

    # Prevent side effects because this function makes inplace changes
    data_copy = data.copy()
    dst_labels = label_dst(data_copy.TIME)
    # We make a column that is unique for all except wintertime
    # This means that in the groupby line, non-to_wintertime
    # lines will never be grouped
    data_copy[tmpcol] = np.where(
        dst_labels == "to_wintertime", -1, np.arange(len(data_copy.index))
    )
    groupcols = data_copy.columns.tolist()
    groupcols.remove("VALUE")  # in place only
    data_copy = data_copy.groupby(groupcols).mean().reset_index().drop(tmpcol, axis=1)

    logging.info(
        "TIME colum changed because of Average_winter_time. "
        "Before it had {} rows, now it has {}".format(data.shape[0], data_copy.shape[0])
    )

    return data_copy
