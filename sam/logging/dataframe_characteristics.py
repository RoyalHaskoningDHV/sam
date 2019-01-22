import logging

logger = logging.getLogger(__name__)


def log_dataframe_characteristics(df):
    """
    Given a dataframe log it's characteristics with the default logger

    Logs:
    - Dimensions
    - Column names
    - Column types

    Parameters
    ----------
    df : A pandas dataframe

    Returns
    -------
    Nothing

    Examples
    --------
    >>> from sam.logging import log_dataframe_characteristics
    >>> log_dataframe_characteristics(df)
    """
    logger.info("columns: %s", df.shape[0])
    logger.info("rows: %s", df.shape[1])

    for i in df.columns.values:
        # Take the first value of a column to get the real type
        logger.info("column: %s, type: %s", i, type(df[i].iat[0]))
