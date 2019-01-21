import pandas as pd


def range_lag_column(original_column, range_shift=(0, 1)):
    """
    Lags a column with a range. Will not lag the actual value,
    but will set a 1 in the specified range for any non-zero value.

    The range can be positive and/or negative. If negative it will 'lag'
    to the future.

    Parameters
    ----------
    original_column: series
                     The original column with non-zero items to lag
    range_shift: tuple (default=(0, 1))
                 The range to lag the original column, it is inclusive.
                 A value of 0 is no lag at all.

    Returns
    -------
    The lagged column as a series

    Example
    -------
    >>> from sam.feature_engineering import range_lag_column
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({"outcome" : [0, 0, 1, 0, 0, 0, 1]})
    >>> df['outcome_lag'] = range_lag_column(df['outcome'], (1, 2))
    >>> df
        outcome outcome_lag
    0   0       1
    1   0       1
    2   1       0
    3   0       0
    4   0       1
    5   0       1
    6   1       0
    """
    original_column = pd.Series(original_column)
    # For loop will fail if not in order
    range_shift = sorted(range_shift)

    df = pd.DataFrame()
    cols = []
    for i in range(range_shift[0], range_shift[1]+1):
        cols.append('lag_' + str(i))
        df['lag_' + str(i)] = original_column.shift(-i)

    # Merge into one outcome column
    return df[cols].any(axis='columns').astype(int)
