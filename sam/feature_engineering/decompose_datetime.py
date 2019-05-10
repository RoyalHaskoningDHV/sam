from sam.logging import log_dataframe_characteristics, log_new_columns
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def decompose_datetime(df, column='TIME', components=[], cyclicals=[], remove_original=True):
    """
    Decomposes a time column to one or more components suitable as features

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
    remove_original: bool
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)

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
    result = df.copy()

    logging.debug("Decomposing datetime, number of dates: {}. Components: ".
                  format(len(result[column]), components))

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
        result = recode_cyclical_features(result, cyclicals, remove_original, column)

    return(result)


def recode_cyclical_features(df, cols, remove_original=True, column=''):
    """
    Convert cyclical features (like day of week, hour of day) to
    continuous variables, so that sunday and monday are close together
    numerically.
    See:
    - https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
    - http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The column names to convert to continuous numerical values
    remove_original: bool
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    column: string
        name of original time column in df (e.g. TIME)

    Returns
    -------
    new_df: pandas dataframe
        The input dataframe with cols removed, and replaced by the
        converted features (2 for each feature).
    """

    # add underscore to column if not empty
    if not column == '':
        column = column + '_'

    # test inputs
    assert isinstance(df, pd.DataFrame), 'df should be pandas dataframe'
    assert isinstance(cols, list), 'cols should be a list of columns to convert'
    assert isinstance(remove_original, bool)

    # save copy of original dataframe
    new_df = df.copy()

    logging.debug("Sine/cosine converting cyclicals columns: %s" % (cols))

    for col in cols:

        # prepend column name (like TIME) to match df column names
        col = column + col

        # test whether column is in dataframe
        assert col in df.columns, '%s is not in input dataframe' % col
        # test colum to convert is of correct data type
        assert df[col].dtype in [int, float], 'cyclical to convert should be integer or float'

        # rescale feature so it runs from 0-2pi:
        norm_feature = 2 * np.pi * df[col] / df[col].max()
        # convert cyclical to 2 variables that are offset:
        new_df[col+'_sin'] = np.sin(norm_feature)
        new_df[col+'_cos'] = np.cos(norm_feature)

        # drop the original
        if remove_original:
            new_df = new_df.drop(col, axis=1)

    # log changes
    log_new_columns(new_df, df)
    log_dataframe_characteristics(new_df, logging.DEBUG)

    return new_df
