import pandas as pd


def complete_timestamps(df, freq='H', start_time='', end_time='',
                        aggregate_method='', fillna_method=''):
    """
    Create a dataframe with all timestamps according to a given frequency
    Giving a dataframe with timestamps such as 9.45, 9.59, 10.10 and a
    frequency of 15 min will return 9.45, 10.00 and 10.15

    Based on start, end and frequency, add rows for missing timestamps
    and fill NA values based on specified function
    NA's can continued from previous etc.

    Parameters
    ----------
    df : pandas dataframe with TIME, ID and VALUE columns, shape = (nrows, 3)
        Dataframe from which the values are created
    start_time : str or datetime-like, optional (default = '')
        the start time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object
    end_time : str or datetime-like, optional (default = '')
        the end time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object
    freq : str or DateOffset, optional (default = 'H')
        the frequency with which the time features are made
        frequencies can have multiples, e.g. "15 min" for 15 minutes
        https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    aggregate_method : string or list, optional (default = '')
        method that is used to aggregate value.
        Can be strings such as mean, sum, min, max.
    fillna_method : string, optional (default = '')
        method used to fill NA values, must follow pandas data frame fillna.
        Can be strings such as ffill, bfill, pad

    Returns
    -------
    complete_df : pandas dataframe, shape (length(TIME) * length(unique IDs), 3)
        dataframe containing all possible combinations of timestamps and IDs
        with selected frequency, aggregate method and fillna method

    Examples
    --------
    >>> from sam.preprocessing import complete_timestamps
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
    >>>                             datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 46)],
    >>>                    'ID': "SENSOR",
    >>>                    'VALUE': [1, 20, 3, 20]})
    >>>
    >>> complete_timestamps(df, freq = "15 min", end_time="2018-06-09 12:15:00",
    >>>                     aggregate_method = "median", fillna_method="ffill")
        TIME                ID      VALUE
    0   2018-06-09 11:00:00 SENSOR  1.0
    1   2018-06-09 11:15:00 SENSOR  1.0
    2   2018-06-09 11:30:00 SENSOR  11.5
    3   2018-06-09 11:45:00 SENSOR  20.0
    4   2018-06-09 12:00:00 SENSOR  20.0
    """
    if df.empty:
        raise ValueError('No dataframe found')

    if not start_time:
        start_time = df['TIME'].min()
    if not end_time:
        end_time = df['TIME'].max()
    
    time, ids = pd.core.reshape.util.cartesian_product(
        [pd.date_range(start=start_time,
                       end=end_time,
                       freq=freq
                       ),
         df['ID'].unique()
         ])

    complete_df = pd.DataFrame(dict(TIME=time, ID=ids))
    complete_df['TIME'] = pd.to_datetime(complete_df['TIME'], dayfirst=True)\
        .dt.floor(freq)
    
    if aggregate_method:
        df = df.groupby([pd.Grouper(key='TIME',
                                    freq=freq), 'ID'
                         ])\
            .agg({'VALUE': aggregate_method})

        df.reset_index(drop=False, inplace=True)

    complete_df = complete_df.merge(df, how='left', on=['TIME', 'ID'])

    if fillna_method:
        complete_df.fillna(method=fillna_method, inplace=True)
        
    return complete_df
