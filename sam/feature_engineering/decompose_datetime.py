import pandas as pd


def decompose_datetime(df, column='TIME', components=[]):
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

    return(result)
