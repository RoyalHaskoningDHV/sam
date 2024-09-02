import logging

import pandas as pd
from sam.logging_functions import log_new_columns

logger = logging.getLogger(__name__)


def sum_grouped_columns(df: pd.DataFrame, sep: str = "#", skipna: bool = True) -> pd.DataFrame:
    """
    Utility function to sum columns together based on groups. The column names are assumed to look
    like groupname#suffix. For example: GROUP1#lag_1_day, or GROUP2#sum_1_week. In these
    examples, the groups are GROUP1 and GROUP2 respectively. This function will find all the
    groups, and sum all the columns in the same group together. If a column does not contain the
    separator `sep`, the entire column name is assumed to be the groupname. This means columns like
    GROUP2 and GROUP2#lag_0 will be assumed to be in the same group.

    This function is mainly useful when dealing with a dataframe filled with shapley values. In
    this case, when many features are in the same group, it may be useful to sum these shapley
    values, to calculate a combined contribution that the entire group has. Keep in mind that
    this has the potential to 'wipe out' shapley values: if GROUP1#lag_0 has a large positive
    contribution, and GROUP1#lag_1 has a large negative contribution, then the group GROUP1 as
    a whole will have a contribution near 0. This is mathematically correct, and does indeed
    mean that GROUP1 as a whole had a very small effect on the prediction.

    Parameters
    ----------
    df: pd.DataFrame
       The dataframe whose columns will be added together
    sep: str, optional (default='#')
       The seperator character. The group of a column is defined as everything before the first
       occurrence of this character
    skipna: boolean, optional (default=True)
       Whether or not to ignore missing values in columns. If true, missing values are treated as
       0. Else, missing values are not ignored and the sum for that particular group/row
       combination will be missing as well.

    Returns
    -------
    summed_df: dataframe
        A dataframe with the same row-index as df, but with less columns. All the columns in the
        same group have been summed together.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'X#lag_0': [1, 2, 3],
    ...    'X#lag_1': [1, 2, 3],
    ...    'Y': [5, 5, 5]
    ... })
    >>> sum_grouped_columns(df)
       X  Y
    0  2  5
    1  4  5
    2  6  5
    """
    logger.debug("Now running sum_grouped_columns with sep={}, skipna={}".format(sep, skipna))
    foo = df.copy()
    groups = foo.columns.str.extract(r"^([^{}]*)".format(sep)).values.reshape(-1)
    if skipna:
        result = foo.groupby(groups, axis=1).sum()
    else:
        result = foo.groupby(groups, axis=1).apply(lambda x: x.sum(skipna=False, axis=1))
    log_new_columns(result, df)
    return result


def has_strictly_increasing_index(df: pd.DataFrame, linear: bool = True) -> bool:
    """
    Utility function to validate the index of a dataframe:
    - Check if it is a DatetimeIndex
    - Check if it is strictly increasing
    - Optional: check if increases are constant over the index

    Parameters
    ----------
    df: dataframe
       The dataframe whose index needs to be checked
    linear: bool, optional (default=True)
       True to also check that the index steps are constant over the whole index length

    Returns
    -------
    bool
        Whether the index of the dataframe is a DatetimeIndex and is strictly monotonic increasing
    """

    def _index_has_constant_increase(df: pd.DataFrame):
        series_diff: pd.Series = df.index.to_series().diff()
        if series_diff.size > 1:
            # disregard first index (always 0), use uniqueness to validate the constant increase
            return False if series_diff[1:].unique().size > 1 else True
        else:
            return False

    if isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex) and (
        df.index.is_monotonic_increasing and df.index.is_unique
    ):
        if linear and not _index_has_constant_increase(df):
            return False
        else:
            return True
    else:
        return False


def make_df_monotonic(df: pd.DataFrame, aggregate_func: str = "max") -> pd.DataFrame:
    """
    Utility function to force monotonicity over the columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the columns over which we want to force monotonicity
    aggregate_func : str, optional
        Parameter to aggregate columns together.
        By default, "max" which results in a monotonic increase over the columns (left to right).
        Set to "min" to force monotonic decrease over the columns (left to right).

    Returns
    -------
    pd.DataFrame
        Dataframe that is now monotonic over the columns.

    Raises
    ------
    ValueError
        If the aggregate_func raises an exception.
    """
    if df.empty:
        return df

    df_copy = df.copy()

    _LEGAL_AGG_FUNC = ("min", "max")
    if aggregate_func not in _LEGAL_AGG_FUNC:
        raise ValueError(
            f"Illegal aggregate_func={aggregate_func}, please choose from {_LEGAL_AGG_FUNC}"
        )

    for idx, col in enumerate(df_copy.columns):
        if idx > 0:
            df_copy[col] = getattr(df_copy.iloc[:, : idx + 1], aggregate_func)(axis=1)

    return df_copy


def contains_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to check if a dataframe contains any NaN values.

    Parameters
    ----------
    df: dataframe
       The dataframe whose index needs to be checked

    Returns
    -------
    bool
        Whether the dataframe contains any NaN values
    """
    return df.isnull().values.any()


def assert_contains_nans(df: pd.DataFrame, msg: str = "Data cannot contain nans") -> None:
    """
    Utility function to check if a dataframe contains any NaN values.

    Parameters
    ----------
    df: dataframe
        The dataframe whose index needs to be checked
    msg: str, optional (default="Data cannot contain nans")
        The error message to raise if the dataframe contains nans
    """
    if contains_nans(df):
        raise ValueError(msg)
