import logging
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setdoc(func: Callable):
    func.__doc__ = """
    This documentation covers `correct_above_threshold`, `correct_below_threshold`
    and `correct_outside_range`. These three functions can be used to filter extreme
    values or fill them with a specified method. The function can correctly handle
    series with a `DatetimeIndex`, to interpolate correctly even in the case of
    measurements with a varying frequency.

    Note: this function does not affect nans. To filter/fill missing values, use `pd.fillna
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html>`_
    instead.

    Parameters
    ----------
    series: A pandas series
         The series containing potential outliers
    threshold: number, (default = 1) or a tuple (default = (0,1))
        The exclusive threshold. A number for above or below, for
        `correct_outside_range` it should be a tuple
    method: string (default = "na")
        To what the threshold exceeding values should be corrected, options are:

        - If 'na', set values to `np.nan`
        - If 'previous', set values to previous non non-exceeding, non-na value
        - If 'average', linearly interpolate values using
            `pandas.DataFrame.interpolate
            <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_,
            *might leak and requires an index*
        - If 'clip': set to the max threshold, lower/upper in case of range
        - If 'value', set to a specific value, specified in `value` parameter
        - If 'remove', removes complete row.
    value: (default = None)
        If `method` is 'value', set the threshold exceeding entry to this value

    Returns
    -------
    series: pandas series
        The original series with the threshold exceeding values corrected

    Examples
    --------
    >>> from sam.preprocessing import correct_below_threshold
    >>> import pandas as pd
    >>> data = pd.Series([0, -1, 2, 1], index = [2, 3, 4, 6])
    >>>
    >>> correct_below_threshold(data, method = "average", threshold=0)
    2    0.0
    3    1.0
    4    2.0
    6    1.0
    dtype: float64
    >>> from sam.preprocessing import correct_outside_range
    >>> import pandas as pd
    >>> data = pd.Series([0, -1, 2, 1])
    >>>
    >>> correct_outside_range(data, method = "na", threshold=(0,1))
    0    0.0
    1    NaN
    2    NaN
    3    1.0
    dtype: float64
    >>> from sam.preprocessing import correct_above_threshold
    >>> import pandas as pd
    >>> data = pd.Series([0, -1, 2, 1])
    >>>
    >>> correct_above_threshold(data, method = "remove", threshold = 1)
    0    0
    1   -1
    3    1
    dtype: int64
    """
    return func


def _fix_values(
    series: pd.Series,
    outliers: pd.Series,
    threshold: Union[float, Tuple[float, float]],
    method: str,
    value: float,
    outside_range: bool = False,
):
    """
    Internal function used to deal with extreme values in `correct_above_threshold`,
    `correct_below_threshold` and `correct_outside_range`. Read the documentation
    from those functions to get more insigns and information regarding the
    input parameters
    """
    methods = ["na", "clip", "previous", "average", "value", "remove"]
    if method not in methods:
        raise ValueError("Method {} not allowed, it must be in {}".format(method, methods))

    logging.debug(
        "Now correcting threshold"
        "threshold={}, method={}, value={}".format(threshold, method, value)
    )

    na_locations = series.isna()

    if method == "previous":
        series.loc[outliers] = np.nan
        series = series.fillna(method="ffill")
        series.loc[na_locations] = np.nan  # Recover original nans
    elif method == "average":
        series.loc[outliers] = np.nan
        # Values seems to work with time indexes as well
        series = series.interpolate(method="values")
        series.loc[na_locations] = np.nan  # Recover original nans
    elif method == "value":
        series.loc[outliers] = value
    elif method == "remove":
        series = series.loc[~outliers]
    elif method == "na":
        series.loc[outliers] = np.nan
    elif method == "clip":
        if outside_range:
            series = series.clip(lower=threshold[0], upper=threshold[1])
        else:
            series.loc[outliers] = threshold

    logger.info(
        "Correct_outside_range changed {} values using method {}".format(sum(outliers), method)
    )
    logger.info(
        "The series previously had {} missing values, now it has {}".format(
            na_locations.sum(), series.isna().sum()
        )
    )

    return series


@setdoc
def correct_above_threshold(series, threshold=1, method="na", value=None):
    outliers = series > threshold
    return _fix_values(series, outliers, threshold, method, value)


@setdoc
def correct_below_threshold(series, threshold=0, method="na", value=None):
    outliers = series < threshold
    return _fix_values(series, outliers, threshold, method, value)


@setdoc
def correct_outside_range(series, threshold=(0, 1), method="na", value=None):
    if not threshold[0] < threshold[1]:
        raise ValueError("Threshold must be a tuple (a, b) with a < b")
    outliers = (series < threshold[0]) | (series > threshold[1])
    return _fix_values(series, outliers, threshold, method, value, outside_range=True)
