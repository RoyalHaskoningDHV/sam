import logging
from sam.logging import log_dataframe_characteristics
import numpy as np

logger = logging.getLogger(__name__)


def setdoc(func):

    func.__doc__ = """
    This documentation covers *correct_above_threshold*, *correct_below_threshold*
    and *correct_outside_range*. These three functions can be used to filter extreme
    values or fill them with a specified method. The function can correctly handle
    dataframes with a DatetimeIndex, to interpolate correctly even in the case of
    measurements with a varying frequency.

    Note: Using a method that is not 'na' will remove 'na' values from the target column

    Parameters
    ----------
    df : A pandas dataframe
         The dataframe with the target_column included
    target_column: string (default = TARGET)
                   The name of the column to check against the threshold
    threshold: number, (default = 1) or a tuple (default = (0,1))
               The exclusive threshold. A number for above or below, for
               correct_outside_range it should be a tuple
    method: string (default = "na")
            To what the threshold exceedingn values should be corrected,
            options are:
            - 'na'
            - 'previous'
            - 'average': linearly interpolated usind pandas.DataFrame.interpolate,
                         *might leak and requires an index*
            - 'clip': set to the max threshold, lower and upper in case of range
            - 'value': set to a specific value, specified in 'value parameter'
            - 'remove': removes complete row.
    value: (default = None)
           If 'method' is 'value', set the threshold exceeding entry to this value

    Returns
    -------
    The original dataframe with the threshold exceeding values corrected

    Examples
    --------
    >>> from sam.preprocessing import correct_below_threshold
    >>> import pandas as pd
    >>> df = pd.DataFrame({"TEST" : [2, 3, 4, 6],
    >>>            "TARGET" : [0, -1, 2, 1]})
    >>>
    >>> df = df.set_index(df['TEST'])
    >>> correct_below_threshold(df, method = "average", threshold=0)
        TEST    TARGET
    TEST
    2   2       0.0
    3   3       1.0
    4   4       2.0
    6   6       1.0
    >>> from sam.preprocessing import correct_outside_range
    >>> import pandas as pd
    >>> df = pd.DataFrame({"TEST" : [2, 3, 4, 6],
    >>>            "TARGET" : [0, -1, 2, 1]})
    >>>
    >>> correct_outside_range(df, method = "na", threshold=(0,1))
        TEST    TARGET
    0   2       0.0
    1   3       NaN
    2   4       NaN
    3   6       1.0
    >>> from sam.preprocessing import correct_above_threshold
    >>> import pandas as pd
    >>> df = pd.DataFrame({"TEST" : [2, 3, 4, 6],
    >>>            "TARGET" : [0, -1, 2, 1]})
    >>>
    >>> correct_above_threshold(df, method = "remove", threshold = 0)
        TEST    TARGET
    0   2       0.0
    1   6       1.0
    """
    return func


def _fix_values(df, target_column, threshold, method, value):
    """
    Helper function, read the other docs
    """
    assert method in ['na', 'clip', 'previous', 'average', 'value', 'remove']
    original_nas = df[target_column].isna().sum()

    logging.debug("Now correcting threshold, target_column={}, "
                  "threshold={}, method={}, value={}".
                  format(target_column, threshold, method, value))

    # Only for range cutoffs we need the original value
    if (method != 'clip'):
        # This fixes case 'if (method == "na")'
        df.loc[df[target_column + '_INCORRECT'], target_column] = np.nan
    else:
        if (type(threshold) == tuple):
            df[target_column] = df[target_column].clip(lower=threshold[0], upper=threshold[1])
        else:
            df[target_column] = df[target_column].where(-(df[target_column + '_INCORRECT']),
                                                        threshold)

    if (method == "previous"):
        df[target_column] = df[target_column].fillna(method='ffill')
    elif (method == "average"):
        # Values seems to work with time indexes as well
        df[target_column] = df[target_column].interpolate(method='values')
    elif (method == 'value'):
        df[target_column] = df[target_column].fillna(value)
    elif (method == 'remove'):
        df = df.dropna(subset=[target_column])

    logger.info("Correct_outside_range changed {} values using method {}".
                format(sum(df[target_column + '_INCORRECT']), method))
    logger.info("The column {} previously had {} missing values, now it has {}".
                format(target_column, original_nas, df[target_column].isna().sum()))

    return df.drop(target_column + '_INCORRECT', axis=1)


@setdoc
def correct_above_threshold(df, target_column="TARGET", threshold=1, method="na", value=None):

    df[target_column + '_INCORRECT'] = df[target_column] > threshold
    return _fix_values(df, target_column, threshold, method, value)


@setdoc
def correct_below_threshold(df, target_column="TARGET", threshold=0, method="na", value=None):
    df[target_column + '_INCORRECT'] = df[target_column] < threshold
    return _fix_values(df, target_column, threshold, method, value)


@setdoc
def correct_outside_range(df, target_column="TARGET", threshold=(0, 1), method="na", value=None):
    df[target_column + '_INCORRECT'] = (df[target_column] < threshold[0]) | \
                                        (df[target_column] > threshold[1])

    return _fix_values(df, target_column, threshold, method, value)
