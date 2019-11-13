from sam.logging import log_dataframe_characteristics, log_new_columns
import numpy as np
import pandas as pd
import logging
import warnings
logger = logging.getLogger(__name__)


def get_maxes_from_strings(cyclicals):
    # This contains cyclical_maxes for pandas datetime features
    # It only contains maxes for those features that are actually cyclical.
    # For example, 'year' is not cyclical so is not included here.
    # Note that the maxes are chosen such that these values are equivalent to 0.
    # e.g.: a minute of 60 is equivalent to a minute of 0
    # For month, dayofyear and week, these are approximations, but close enough.
    lookup = {
        "day": 31,
        "dayofweek": 7,
        "weekday": 7,
        "dayofyear": 366,
        "hour": 24,
        "microsecond": 1e6,
        "minute": 60,
        "month": 12,
        "quarter": 4,
        "second": 60,
        "week": 53
    }
    for c in cyclicals:
        if c not in lookup:
            raise ValueError(str(c) + " is not a known cyclical. "
                                      "Provide cyclical_maxes yourself")
    return [lookup[c] for c in cyclicals]


def decompose_datetime(df, column='TIME', components=[], cyclicals=[], remove_original=None,
                       remove_categorical=True, keep_original=True, cyclical_maxes=None,
                       cyclical_mins=0):
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
        Name of the source column to extract components from. Should have a datetime format
    components: list
        List of components to extract from datatime column. All default pandas dt components are
        supported, and some custom functions will be implemented in the future.
    cyclicals: list
        Newly created .dt time variables (like hour, month) you want to convert
        to cyclicals using sine and cosine transformations.
        Cyclicals are variables that do not inncrease linearly, but wrap around.
        such as days of the week and hours of the day.
        Format is identical to components input.
    remove_original: bool, optional (default=None)
        Deprecated. Will be removed in a future release. Please use remove_categorical instead.
    remove_categorical: bool, optional (default=True)
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original ones
    cyclical_maxes: list, optional (default=None)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.
    cyclical_mins: list, optional (default=0)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.

    Returns
    -------
    result : dataframe
        The original dataframe with extra columns containing time components

    Examples
    --------
    >>> from sam.feature_engineering import decompose_datetime
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({'TIME': pd.date_range("2018-12-27", periods = 4),
    >>>                    'OTHER_VALUE': [1, 2, 3,2]})
    >>>
    >>> decompose_datetime(df, components= ["year", "weekday_name"])
        TIME        OTHER_VALUE TIME_year   TIME_weekday_name
    0   2018-12-27  1           2018        Thursday
    1   2018-12-28  2           2018        Friday
    2   2018-12-29  3           2018        Saturday
    3   2018-12-30  2           2018        Sunday
    """
    if remove_original is not None:
        msg = ("the remove_original parameter in decompose_datetime is deprecated."
               "Please use the remove_categorical parameter instead.")
        warnings.warn(msg, DeprecationWarning)
        remove_categorical = remove_original

    if keep_original:
        result = df.copy()
    else:
        result = pd.DataFrame(index=df.index)

    logging.debug("Decomposing datetime, number of dates: {}. Components: ".
                  format(len(df[column]), components))

    # We should check first if the column has a compatible type
    pandas_functions = [f for f in dir(df[column].dt) if not f.startswith('_')]

    custom_functions = []
    # Iterate the requested components
    for component in components:
        # Check if this is a default pandas functionality
        if component in pandas_functions:
            result[column + '_' + component] = getattr(df[column].dt, component)
        elif component in custom_functions:
            # Here we will apply custom functions
            pass
        else:
            raise NotImplementedError("Component %s not implemented" % component)

    # do this before converting to cyclicals, as this has its own loggin:
    log_new_columns(result, df)
    log_dataframe_characteristics(result, logging.DEBUG)

    # convert cyclicals
    assert isinstance(cyclicals, list), 'cyclicals must be of type list'
    if cyclicals != []:
        result = recode_cyclical_features(result, cyclicals, column=column,
                                          remove_categorical=remove_categorical,
                                          cyclical_maxes=cyclical_maxes,
                                          cyclical_mins=cyclical_mins,
                                          keep_original=True)

    return(result)


def recode_cyclical_features(df, cols, remove_original=None, column='', remove_categorical=True,
                             keep_original=True, cyclical_maxes=None, cyclical_mins=0):
    """
    Convert cyclical features (like day of week, hour of day) to
    continuous variables, so that sunday and monday are close together
    numerically.
    See:
    - https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
    - http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

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
    remove_original: bool, optional (default=None)
        Deprecated. Will be removed in a future release. Please use remove_categorical instead.
    column: string, optional (default='')
        name of original time column in df (e.g. TIME)
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
    cyclical_mins: array-like, optional (default=0)
        The minimums that your data can reach. Keep in mind that the maximum value and the
        minimum value will be encoded as the same value. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    new_df: pandas dataframe
        The input dataframe with cols removed, and replaced by the
        converted features (2 for each feature).
    """
    if remove_original is not None:
        msg = ("the remove_original parameter in recode_cyclical_features is deprecated."
               "Please use the remove_categorical parameter instead.")
        warnings.warn(msg, DeprecationWarning)
        remove_categorical = remove_original

    # add underscore to column if not empty
    if not column == '':
        column = column + '_'

    # test inputs
    assert isinstance(df, pd.DataFrame), 'df should be pandas dataframe'
    assert isinstance(cols, list), 'cols should be a list of columns to convert'
    assert isinstance(remove_categorical, bool)

    if cyclical_maxes is None:
        cyclical_maxes = get_maxes_from_strings(cols)

    if np.isscalar(cyclical_mins):
        cyclical_mins = [cyclical_mins] * len(cyclical_maxes)

    # save copy of original dataframe
    if keep_original:
        new_df = df.copy()
    # We don't want to keep the orignal columns, but we want to keep the categoricals
    elif not remove_categorical:
        new_df = df.copy()[[column + col for col in cols]]
    else:
        new_df = pd.DataFrame(index=df.index)

    logging.debug("Sine/cosine converting cyclicals columns: %s" % (cols))

    for cyclical_min, cyclical_max, col in zip(cyclical_mins, cyclical_maxes, cols):

        if cyclical_min >= cyclical_max:
            raise ValueError("Cyclical min {} is higher than cyclical max {} for column {}"
                             .format(cyclical_min, cyclical_max, col))

        # prepend column name (like TIME) to match df column names
        col = column + col

        # test whether column is in dataframe
        assert col in df.columns, '%s is not in input dataframe' % col

        # rescale feature so it runs from 0 to 2*pi:
        # Features that exceed the maximum are rolled over by the sinus/cosinus:
        # e.g. if min=0 max=7, 9 will be treated the same as 2
        norm_feature = (df[col] - cyclical_min) / (cyclical_max - cyclical_min)
        norm_feature = 2 * np.pi * norm_feature
        # convert cyclical to 2 variables that are offset:
        new_df[col+'_sin'] = np.sin(norm_feature)
        new_df[col+'_cos'] = np.cos(norm_feature)

        # drop the original. if keep_original is False, this is unneeded: it was already removed
        if remove_categorical and keep_original:
            new_df = new_df.drop(col, axis=1)

    # log changes
    log_new_columns(new_df, df)
    log_dataframe_characteristics(new_df, logging.DEBUG)

    return new_df
