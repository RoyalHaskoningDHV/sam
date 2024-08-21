import datetime
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from sam.logging_functions import log_dataframe_characteristics, log_new_columns

logger = logging.getLogger(__name__)


@dataclass
class CyclicalMaxes:
    """Class for keeping track of maximum integer values of specific cyclicals"""

    day: int = 31
    dayofweek: int = 7
    weekday: int = 7
    dayofyear: int = 366
    hour: int = 24
    microsecond: int = 1000000
    minute: int = 60
    month: int = 12
    quarter: int = 4
    second: int = 60
    week: int = 53
    secondofday: int = 86400

    @classmethod
    def get_maxes_from_strings(cls, cyclicals: Sequence[str]) -> List[int]:
        """
        This method retrieves cyclical_maxes for pandas datetime features
        The CyclicalMaxes class contains maxes for those features that are actually cyclical.
        For example, 'year' is not cyclical so is not included here.
        Note that the maxes are chosen such that these values are equivalent to 0.
        e.g.: a minute of 60 is equivalent to a minute of 0
        For month, dayofyear and week, these are approximations, but close enough.

        Parameters
        ----------
        cyclicals : list
            List of cyclical strings that match attributes in self class

        Returns
        -------
        list
            List of integer representations of the cyclicals
        """
        for c in cyclicals:
            if c not in cls.__annotations__:
                raise ValueError(
                    str(c) + " is not a known cyclical, please " "provide cyclical_maxes yourself."
                )
        return [getattr(cls, c.lower()) for c in cyclicals]


def decompose_datetime(
    df: pd.DataFrame,
    column: Optional[str] = "TIME",
    components: Optional[Sequence[str]] = None,
    cyclicals: Optional[Sequence[str]] = None,
    onehots: Optional[Sequence[str]] = None,
    remove_categorical: bool = True,
    keep_original: bool = True,
    cyclical_maxes: Optional[Sequence[int]] = None,
    cyclical_mins: Optional[Union[Sequence[int], int]] = (0,),
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Decomposes a time column to one or more components suitable as features.

    The input is a dataframe with a pandas timestamp column. New columns will be added to this
    dataframe. For example, if column is 'TIME', and components is ['hour', 'minute'], two
    columns: 'TIME_hour' and 'TIME_minute' will be added.

    Optionally, cyclical features can be added instead. For example, if cyclicals is ['hour'],
    then the 'TIME_hour' column will not be added, but two columns 'TIME_hour_sin' and
    'TIME_hour_cos' will be added instead. If you want both the categorical and cyclical features,
    set 'remove_categorical' to False.

    Parameters
    ----------
    df: dataframe
        The dataframe with source column
    column: str (default='TIME')
        Name of the source column to extract components from. Note: time column should have a
        datetime format. if None, it is assumed that the TIME column will be the index.
    components: list
        List of components to extract from datatime column. All default pandas dt components are
        supported, and some custom functions: `['secondofday', 'week']`.
        Note: `week` was added here since it is deprecated in pandas in favor of
        `isocalendar().week`
    cyclicals: list
        List of strings of newly created .dt time variables (like hour, month) you want to convert
        to cyclicals using sine and cosine transformations. Cyclicals are variables that do not
        increase linearly, but wrap around, such as days of the week and hours of the day.
        Format is identical to `components` input.
    onehots: list
        List of strings of newly created .dt time variables (like hour, month) you want to convert
        to one-hot-encoded variables. This is suitable when you think that variables do not
        vary smoothly with time (e.g. Sunday and Monday are quite different).
        This list must be mutually exclusive from cyclicals, i.e. non-overlapping.
    remove_categorical: bool, optional (default=True)
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original ones
    cyclical_maxes: sequence, optional (default=None)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.
    cyclical_mins: sequence or int, optional (default=0)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.
    timezone: str, optional (default=None)
        if tz is not None, convert the time to the specified timezone, before creating features.
        timezone can be any string that is recognized by pytz, for example `Europe/Amsterdam`.
        We assume that the TIME column is always in UTC,
        even if the datetime object has no tz info.
    Returns
    -------
    dataframe
        The original dataframe with extra columns containing time components

    Examples
    --------
    >>> from sam.feature_engineering import decompose_datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': pd.date_range("2018-12-27", periods = 4),
    ...                    'OTHER_VALUE': [1, 2, 3, 2]})
    >>> decompose_datetime(df, components= ["year", "dayofweek"])
            TIME  OTHER_VALUE  TIME_year  TIME_dayofweek
    0 2018-12-27            1       2018               3
    1 2018-12-28            2       2018               4
    2 2018-12-29            3       2018               5
    3 2018-12-30            2       2018               6
    """
    components = [] if components is None else components
    cyclicals = [] if cyclicals is None else cyclicals
    onehots = [] if onehots is None else onehots
    cyclical_maxes = [] if cyclical_maxes is None else cyclical_maxes

    if np.any([c in cyclicals for c in onehots]):
        raise ValueError("cyclicals and onehots are not mutually exclusive")

    if keep_original:
        result = df.copy()
    else:
        result = pd.DataFrame(index=df.index)

    if column is None:
        timecol = df.index.to_series().copy()
        column = "" if timecol.name is None else timecol.name
    else:
        timecol = df[column].copy()

    logging.debug(
        f"Decomposing datetime, number of dates: {len(timecol)}. " f"Components: {components}"
    )

    # Fix timezone
    if timezone is not None:
        if timecol.dt.tz is not None:
            if timecol.dt.tz != datetime.timezone.utc:
                raise ValueError(
                    "Data should either be in UTC timezone or it should have no"
                    " timezone information (assumed to be in UTC)"
                )
        else:
            timecol = timecol.dt.tz_localize("UTC")
        timecol = timecol.dt.tz_convert(timezone)

    result = _create_time_cols(result, components, timecol, column)

    # do this before converting to cyclicals, as this has its own logging:
    log_new_columns(result, df)
    log_dataframe_characteristics(result, logging.DEBUG)

    # convert cyclicals
    if not isinstance(cyclicals, Sequence):
        raise TypeError("cyclicals must be a sequence type")
    if cyclicals:
        result = recode_cyclical_features(
            result,
            cyclicals,
            prefix=column,
            remove_categorical=remove_categorical,
            cyclical_maxes=cyclical_maxes,
            cyclical_mins=cyclical_mins,
            keep_original=True,
        )
    if onehots:
        result = recode_onehot_features(
            result,
            onehots,
            prefix=column,
            remove_categorical=remove_categorical,
            onehot_maxes=cyclical_maxes,
            onehot_mins=cyclical_mins,
            keep_original=True,
        )

    return result


def recode_cyclical_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    prefix: str = "",
    remove_categorical: bool = True,
    keep_original: bool = True,
    cyclical_maxes: Optional[Sequence[int]] = None,
    cyclical_mins: Optional[Union[Sequence[int], int]] = (0,),
) -> pd.DataFrame:
    """
    Convert cyclical features (like day of week, hour of day) to continuous variables, so that
    Sunday and Monday are close together numerically.

    IMPORTANT NOTE: This function requires a global maximum and minimum for the data. For example,
    for minutes, the global maximum and minimum are 0 and 60 respectively, even if your data never
    reaches these global minimums/maximums explicitly. This function assumes that the minimum and
    maximum should be encoded as the same value: minute 0 and minute 60 mean the same thing.

    If you only use cyclical pandas timefeatures, nothing needs to be done. For these features,
    the minimum/maximum will be chosen automatically. These are: ['day', 'dayofweek', 'weekday',
    'dayofyear', 'hour', 'microsecond', 'minute', 'month', 'quarter', 'second', 'week']

    For any other scenario, global minimums/maximums will need to be passed in the parameters
    `cyclical_maxes` and `cyclical_mins`. Minimums are set to 0 by default, meaning that
    only the maxes need to be chosen as the value that is `equivalent` to 0.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to continuous numerical values.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    column: string, optional (default='')
        name of original time column in df, e.g. TIME.
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool, optional (default=True)
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardless of
        this argument.
    cyclical_maxes: array-like, optional (default=None)
        The maximums that your data can reach. Keep in mind that the maximum value and the
        minimum value will be encoded as the same value. By default, None means that only
        standard pandas timefeatures will be encoded.
    cyclical_mins: array-like or scalar, optional (default=[0])
        The minimums that your data can reach. Keep in mind that the maximum value and the
        minimum value will be encoded as the same value. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    dataframe
        The input dataframe with cols removed, and replaced by the converted features (two for
        each feature).
    """

    new_df, prefix, cyclical_maxes, cyclical_mins = _validate_and_prepare_components(
        df=df,
        cols=cols,
        column=prefix,
        remove_categorical=remove_categorical,
        keep_original=keep_original,
        component_maxes=cyclical_maxes,
        component_mins=cyclical_mins,
    )

    logging.debug(f"Sine/cosine converting cyclicals columns: {cols}")

    for cyclical_min, cyclical_max, col in zip(cyclical_mins, cyclical_maxes, cols):
        if cyclical_min >= cyclical_max:
            raise ValueError(
                "Cyclical min {} is higher than cyclical max {} for column {}".format(
                    cyclical_min, cyclical_max, col
                )
            )

        # prepend column name (like TIME) to match df column names
        col = prefix + col

        if col not in df.columns:
            raise ValueError(f"{col} is not in input dataframe")

        # rescale feature so it runs from 0 to 2*pi:
        # Features that exceed the maximum are rolled over by the sine/cosine:
        # e.g. if min=0 and max=7, 9 will be treated the same as 2
        norm_feature: pd.Series = (df[col] - cyclical_min) / (cyclical_max - cyclical_min)
        norm_feature = 2 * np.pi * norm_feature
        # convert cyclical to 2 variables that are offset:
        new_df[col + "_sin"] = np.sin(norm_feature)
        new_df[col + "_cos"] = np.cos(norm_feature)

        # drop the original. if keep_original is False, this is unneeded: it was already removed
        if remove_categorical and keep_original:
            new_df = new_df.drop(col, axis=1)

    # log changes
    log_new_columns(new_df, df)
    log_dataframe_characteristics(new_df, logging.DEBUG)

    return new_df


def recode_onehot_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    prefix: str = "",
    remove_categorical: bool = True,
    keep_original: bool = True,
    onehot_maxes: Optional[Sequence[int]] = None,
    onehot_mins: Optional[Union[Sequence[int], int]] = (0,),
) -> pd.DataFrame:
    """
    Convert time features (like day of week, hour of day) to onehot variables (1 or 0 for each
    unique value).

    IMPORTANT NOTE: This function requires a global maximum and minimum for the data. For example,
    for minutes, the global maximum and minimum are 0 and 60 respectively, even if your data never
    reaches these global minimums/maximums explicitly. Make sure these variables will all be added
    in your onehot columns, otherwise your columns in train and test set could be unmatching.

    If you only use cyclical pandas timefeatures, nothing needs to be done. For these features,
    the minimum/maximum will be chosen automatically. These are: ['day', 'dayofweek', 'weekday',
    'dayofyear', 'hour', 'microsecond', 'minute', 'month', 'quarter', 'second', 'week']

    For any other scenario, global minimums/maximums will need to be passed in the parameters
    `cyclical_maxes` and `cyclical_mins`. Minimums are set to 0 by default, meaning that
    only the maxes need to be chosen as the value that is `equivalent` to 0.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to onehot variables.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    prefix: string, optional (default='')
        name of original time column in df, e.g. 'TIME'.
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool, optional (default=True)
        whether to keep the original time features (i.e. day)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardles of
        this argument.
    onehot_maxes: array-like, optional (default=None)
        The maximums that your data can reach. By default, None means that only
        standard pandas timefeatures will be encoded.
    onehot_mins: array-like or scalar, optional (default=[0])
        The minimums that your data can reach. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    pandas dataframe
        The input dataframe with cols removed, and replaced by the converted features.
    """

    new_df, prefix, onehot_maxes, onehot_mins = _validate_and_prepare_components(
        df=df,
        cols=cols,
        column=prefix,
        remove_categorical=remove_categorical,
        keep_original=keep_original,
        component_maxes=onehot_maxes,
        component_mins=onehot_mins,
    )

    logging.debug(f"onehot converting time columns: {cols}")

    for onehot_min, onehot_max, col in zip(onehot_mins, onehot_maxes, cols):
        col = prefix + col

        if col not in df.columns:
            raise ValueError(f"{col} is not in input dataframe")

        # get the onehot encoded dummies
        dummies: pd.DataFrame = pd.get_dummies(df[col], prefix=col)

        # fill in the weekdays not in the dataset
        for i in range(onehot_min, onehot_max):
            if not "%s_%d" % (col, i) in dummies.columns:
                dummies["%s_%d" % (col, i)] = 0
        dummies_sorted = dummies[np.sort(dummies.columns)].astype("int32")
        new_df = new_df.join(dummies_sorted)

        # drop the original. if keep_original is False, this is unneeded: it was already removed
        if remove_categorical and keep_original:
            new_df = new_df.drop(col, axis=1)

    log_new_columns(new_df, df)
    log_dataframe_characteristics(new_df, logging.DEBUG)

    return new_df


def _create_time_cols(
    df: pd.DataFrame, components: Sequence[str], timecol: pd.Series, prefix: str = ""
) -> pd.DataFrame:
    """Helper function to create all the neccessary time columns

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe in SAM format
    components : Sequence[str]
        Time components that need to be added to the data
    timecol: Series
        A pandas series containing the datetimes, used for making the time columns
    prefix : str
        Prefix of the newly created columns, usually the same as the original time column

    Returns
    -------
    pd.DataFrame
        The dataframe, which includes the time components

    Raises
    ------
    NotImplementedError
        In case the time components are not recognized by pandas or by SAM
    """

    pandas_functions = [f for f in dir(timecol.dt) if not f.startswith("_")]

    custom_functions = ["secondofday", "week"]
    for component in components:
        if component in custom_functions:
            if component == "week":
                df[prefix + "_" + component] = timecol.dt.isocalendar().week
            elif component == "secondofday":
                sec_in_min = 60
                sec_in_hour: int = sec_in_min * 60
                df[prefix + "_" + component] = (
                    timecol.dt.hour * sec_in_hour
                    + timecol.dt.minute * sec_in_min
                    + timecol.dt.second
                )
        elif component in pandas_functions:
            df[prefix + "_" + component] = getattr(timecol.dt, component)
        else:
            raise NotImplementedError(f"Component {component} not implemented")
    return df


def _validate_and_prepare_components(
    df: pd.DataFrame,
    cols: Sequence[str],
    column: str,
    remove_categorical: bool,
    keep_original: bool,
    component_maxes: Optional[Sequence[int]],
    component_mins: Optional[Union[Sequence[int], int]],
) -> Tuple[pd.DataFrame, str, Sequence[int], Sequence[int]]:
    """
    Validates and prepares the dataframe, component (onehot or cyclical) parameters and min/max
    component bounds.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to onehot variables.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    column: string
        name of original time column in df (e.g. TIME)
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool
        whether to keep the original time features (i.e. day)
    keep_original: bool
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardles of
        this argument.
    component_maxes: array-like
        The maximums that your data can reach. By default, None means that only
        standard pandas timefeatures will be encoded.
    component_mins: array-like or scalar
        The minimums that your data can reach. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    new_df: pandas dataframe
        Dataframe with/without the categoricals and the original columns depending on setttings.
    column: string
        updated name of original time column in df, e.g. 'TIME_'.
    component_maxes: array-like
        The maximums that your data can reach.
    component_mins: array-like
        The minimums that your data can reach.
    """

    column = column + "_" if column != "" else column
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be pandas dataframe")
    if not isinstance(cols, Sequence):
        raise TypeError("cols should be a sequence of columns to convert")
    if not isinstance(remove_categorical, bool):
        raise TypeError("remove_categorical should be a boolean")

    if component_maxes is None or not component_maxes:
        component_maxes = CyclicalMaxes.get_maxes_from_strings(cols)

    if np.isscalar(component_mins):
        component_mins = [cast(int, component_mins)] * len(component_maxes)
    elif isinstance(component_mins, (Sequence, np.ndarray)):
        if len(component_mins) == 1:
            component_mins = list(component_mins)
            component_mins *= len(component_maxes)
    else:
        raise TypeError("`component_maxes` needs to be a scalar or array-like")

    if keep_original:
        new_df = df.copy()
    elif not remove_categorical:
        # We don't want to keep the original columns, but we want to keep the categoricals
        new_df = df.copy()[[column + col for col in cols]]
    else:
        new_df = pd.DataFrame(index=df.index)

    return new_df, column, component_maxes, component_mins
