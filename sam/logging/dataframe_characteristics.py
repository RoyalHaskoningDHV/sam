import logging

logger = logging.getLogger(__name__)


def log_dataframe_characteristics(df, level=logging.INFO):
    """
    Given a dataframe log it's characteristics with the default logger

    Logs:
    - Dimensions
    - Column names
    - Column types

    Parameters
    ----------
    df : A pandas dataframe
    level : Loglevel to log at. By default, logging.INFO

    Returns
    -------
    Nothing

    Examples
    --------
    >>> from sam.logging import log_dataframe_characteristics
    >>> log_dataframe_characteristics(df)
    """
    logger.log(level, "columns: %s", df.shape[0])
    logger.log(level, "rows: %s", df.shape[1])
    if df.shape[0] == 0 or df.shape[1] == 0:
        logger.log(level, "No type information of columns, because there were no values")
    else:
        for i in df.columns.values:
            # Take the first value of a column to get the real type
            logger.log(level, "column: %s, type: %s", i, type(df[i].iat[0]))
