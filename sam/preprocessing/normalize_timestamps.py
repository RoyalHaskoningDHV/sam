import pandas as pd
import logging
logger = logging.getLogger(__name__)


def normalize_timestamps(df, freq='H', start_time='', end_time='',
                         aggregate_method='mean', fillna_method=None):
    """
    Create a dataframe with all timestamps according to a given frequency,
    aggregating the values using a specified method (default: 'mean')
    Giving a dataframe with timestamps such as 9.45, 9.59, 10.10 and a
    frequency of 15 min will return 9.45, 10.00 and 10.15

    Based on start, end and frequency, add rows for missing timestamps
    and fill NA values based on specified function, defined using fillna_method
    For example: NA's can be continued from previous timestamps using 'ffill'.

    Parameters
    ----------
    df: pandas dataframe with TIME, ID and VALUE columns, shape = (nrows, 3)
        Dataframe from which the values are created

    start_time: str or datetime-like, optional (default = '')
        the start time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object

    end_time: str or datetime-like, optional (default = '')
        the end time of the period to create features over
        if string, the format 'YYYY/MM/DD HH:mm:SS' will always work
        Pandas also accepts other formats, or a datetime object

    freq: str or DateOffset, optional (default = 'H')
        the frequency with which the time features are made
        frequencies can have multiples, e.g. "15 min" for 15 minutes
        https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    aggregate_method: function, string, dictionary, list of string/functions (default = 'mean')
        Method that is used to aggregate values when multiple values fall
        within a specified frequency region.
        For example, when you have data per 5 minutes, but you're creating a
        an hourly frequency, the values need to be aggregated.
        Can be strings such as mean, sum, min, max, or a function.
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.aggregate.html

    fillna_method: string, optional (default = None)
        Method used to fill NA values, must follow pandas data frame fillna.
        Options are: 'backfill', 'bfill', 'pad', 'ffill', None
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html

    Returns
    -------
    complete_df: pandas dataframe, shape (length(TIME) * length(unique IDs), 3)
        dataframe containing all possible combinations of timestamps and IDs
        with selected frequency, aggregate method and fillna method

    Examples
    --------
    >>> from sam.preprocessing import normalize_timestamps
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
    >>>                             datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 46)],
    >>>                    'ID': "SENSOR",
    >>>                    'VALUE': [1, 20, 3, 20]})
    >>>
    >>> normalize_timestamps(df, freq = "15 min", end_time="2018-06-09 12:15:00",
    >>>                     aggregate_method = "median", fillna_method=None)
        TIME                    ID      VALUE
    0 	2018-06-09 11:00:00 	SENSOR 	1.0
    1 	2018-06-09 11:15:00 	SENSOR 	NaN
    2 	2018-06-09 11:30:00 	SENSOR 	11.5
    3 	2018-06-09 11:45:00 	SENSOR 	20.0
    4 	2018-06-09 12:00:00 	SENSOR 	NaN

    >>> from sam.preprocessing import normalize_timestamps
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
    >>>                             datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 46)],
    >>>                    'ID': "SENSOR",
    >>>                    'VALUE': [1, 20, 3, 20]})
    >>>
    >>> normalize_timestamps(df, freq = "15 min", end_time="2018-06-09 12:15:00",
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

    original_rows = df.shape[0]
    original_nas = df['VALUE'].isna().sum()

    fillna_options = ['backfill', 'bfill', 'pad', 'ffill', None]
    if fillna_method not in fillna_options:
        raise ValueError('fillna_method not in {}'.format(str(fillna_options)))

    if not start_time:
        start_time = df['TIME'].min()
    if not end_time:
        end_time = df['TIME'].max()

    logger.debug("Completing timestamps: freq={}, start_time={}, end_time={}, "
                 "aggregate_method={}, fillna_method={}".format(
                     freq, start_time, end_time, aggregate_method, fillna_method))

    time, ids = pd.core.reshape.util.cartesian_product(
        [pd.date_range(start=start_time,
                       end=end_time,
                       freq=freq
                       ),
         df['ID'].unique()
         ])

    complete_df = pd.DataFrame(dict(TIME=time, ID=ids), columns=['TIME', 'ID'])
    complete_df['TIME'] = pd.to_datetime(complete_df['TIME'], dayfirst=True)\
        .dt.floor(freq)

    # Function currently groups based on first left matching frequency,
    # can be set to the first right frequency within the Grouper function
    df = df.groupby([pd.Grouper(key='TIME',
                                freq=freq),
                     'ID'
                     ])\
        .agg({'VALUE': aggregate_method})

    df = df.reset_index(drop=False)

    complete_df = complete_df.merge(df, how='left', on=['TIME', 'ID'])

    logger.debug("Number of missings before fillna: {}".format(complete_df['VALUE'].isna().sum()))

    if fillna_method:
        complete_df['VALUE'] = complete_df.groupby('ID')['VALUE']\
            .apply(lambda x: x.fillna(method=fillna_method))

    logger.info("Dataframe changed because of normalize_timestamps: "
                "Previously it had {} rows, now it has {}".
                format(original_rows, complete_df.shape[0]))
    logger.info("Also, the VALUE column previously had {} missing values, now it has {}".
                format(original_nas, complete_df["VALUE"].isna().sum()))

    return complete_df
