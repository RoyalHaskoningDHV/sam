import logging

import pandas as pd

logger = logging.getLogger(__name__)


def range_lag_column(
    original_column: pd.Series,
    range_shift: tuple = (0, 1),
) -> pd.Series:
    """
    Lags a column with a range. Will not lag the actual value,
    but will set a 1 in the specified range for any non-zero value.

    The range can be positive and/or negative. If negative it will 'lag'
    to the future.

    Parameters
    ----------
    original_column: pandas series
                     The original column with non-zero items to lag
    range_shift: tuple (default=(0, 1))
                 The range to lag the original column, it is inclusive.
                 A value of 0 is no lag at all.

    Returns
    -------
    pandas series
        The lagged column as a series. The input will be converted to float64.

    Example
    -------
    >>> from sam.feature_engineering import range_lag_column
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({"outcome" : [0, 0, 1, 0, 0, 0, 1]})
    >>> df['outcome_lag'] = range_lag_column(df['outcome'], (1, 2))
    >>> df
       outcome  outcome_lag
    0        0          1.0
    1        0          1.0
    2        1          0.0
    3        0          0.0
    4        0          1.0
    5        0          1.0
    6        1          0.0
    """
    original_column = pd.Series(original_column)
    # For loop will fail if not in order
    range_shift = sorted(range_shift)
    # Window size of the shift (+1 because inclusive)
    window_size_inclusive: int = range_shift[1] - range_shift[0] + 1

    logger.debug(
        "Now lagging range column with length: {}. Range shift: {}".format(
            original_column.size, range_shift
        )
    )

    # Reverse because we want to lag.
    # Then, we take the max which is the boolean version of 'any'
    # At the end, reverse back, which will maintain the index
    result = (
        original_column[::-1]
        .rolling(window_size_inclusive, min_periods=1)
        .max()
        .shift(range_shift[0])
        .fillna(0)[::-1]
    )

    if range_shift[0] < 0:
        result[0 : -range_shift[0]] = (
            original_column[0 : -range_shift[0]]
            .rolling(window_size_inclusive, min_periods=1)
            .max()
        )

    return result
