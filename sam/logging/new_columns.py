import logging

logger = logging.getLogger(__name__)


def log_new_columns(new, old, level=logging.INFO, log_no_changes=False):
    """
    Given a new and old pandas dataframe, checks what columns were removed/added, and logs them

    Logs what columns were removed (if there were any)
    Logs what columns were added (if there were any)
    Only logs column names. Doesn't check content of columns
    By default, logs if there were no changes, but this can be turned off.

    Parameters
    ----------
    new : The new pandas dataframe
    old : The old pandas dataframe
    level : The loglevel. Is logging.INFO by default
    log_no_changes : boolean, if it should be logged when columns are identical. False by default


    Returns
    -------
    Nothing

    Examples
    --------
    >>> from sam.logging import log_new_columns
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
