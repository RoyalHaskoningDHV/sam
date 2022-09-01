import logging

import pandas as pd

logger = logging.getLogger(__name__)


def log_dataframe_characteristics(df: pd.DataFrame, level=logging.INFO):
    """
    Given a dataframe log it's characteristics with the default logger

    Logs:
    - Dimensions
    - Column names
    - Column types

    Parameters
    ----------
    df: pd.DataFrame
        Any pandas dataframe
    level: loglevel (int) (default=logging.INFO)
        Loglevel to log at.

    Returns
    -------
    Nothing

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.data_sources import read_knmi
    >>>
    >>> df = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    ...                  variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    ...                             'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> from sam.logging_functions import log_dataframe_characteristics
    >>> log_dataframe_characteristics(df)
    """
    logger.log(level, f"columns: {df.shape[0]}")
    logger.log(level, f"rows: {df.shape[1]}")
    if df.shape[0] == 0 or df.shape[1] == 0:
        logger.log(level, "No type information of columns, because there were no values")
    else:
        for i in df.columns.values:
            # Take the first value of a column to get the real type
            logger.log(level, f"column: {i}, type: {type(df[i].iat[0])}")
