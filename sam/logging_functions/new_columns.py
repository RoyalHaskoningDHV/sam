import logging

import pandas as pd

logger = logging.getLogger(__name__)


def log_new_columns(
    new: pd.DataFrame,
    old: pd.DataFrame,
    level: int = logging.INFO,
    log_no_changes: bool = False,
):
    """
    Given two pandas dataframe (new and old), checks what columns were removed/added, and logs them

    Logs what columns were removed (if there were any)
    Logs what columns were added (if there were any)
    Only logs column names. Doesn't check content of columns
    By default, logs if there were no changes, but this can be turned off.

    Parameters
    ----------
    new: pd.DataFrame
        The new pandas dataframe
    old: pd.DataFrame
        The old pandas dataframe
    level: loglevel (int) (default=logging.INFO)
        Loglevel to log at.
    log_no_changes : bool (default=False)
        if it should be logged when columns are identical.


    Returns
    -------
    Nothing

    Examples
    --------
    >>> from sam.logging_functions import log_new_columns
    >>> df = pd.DataFrame({'A': [1,2,3], 'C': [4,3,2]})
    >>> new_df = df.rename(columns = {'A': 'B'})
    >>> log_new_columns(new_df, df)
    """
    new_columns = list(set(new.columns.values) - set(old.columns.values))
    removed_columns = list(set(old.columns.values) - set(new.columns.values))

    if len(removed_columns) > 0:
        logger.log(level, "Removed columns: {}".format(removed_columns))
    if len(new_columns) > 0:
        logger.log(level, "Added new columns: {}".format(new_columns))
    if len(removed_columns) == 0 and len(new_columns) == 0 and log_no_changes:
        logger.log(level, "No changes in columns")
