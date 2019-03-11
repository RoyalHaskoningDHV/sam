from sam.logging import log_new_columns
import logging
logger = logging.getLogger(__name__)


def sum_grouped_columns(df, sep='#', skipna=True):
    """
    Utility function to sum columns together based on groups. The column names are assumed to look
    like groupname#suffix. For example: DEBIET#lag_1_day, or INFLUENT#sum_1_week. In these
    examples, the groups are DEBIET and INFLUENT respectively. This function will find all the
    groups, and sum all the columns in the same group together. If a column does not contain the
    '#' character, the entire column name is assumed to be the groupname. This means columns like
    INFLUENT and INFLUENT#lag_0 will be assumed to be in the same group.

    This function is mainly useful when dealing with a dataframe filled with shapley values. In
    this case, when many features are in the same group, it may be useful to sum these shapley
    values, to calculate a combined contribution that the entire group has. Keep in mind that
    this has the potential to 'wipe out' shapley values: if DEBIET#lag_0 has a large positive
    contribution, and DEBIET#lag_1 has a large negative contribution, then the group DEBIET as
    a whole will have a contribution near 0. This is mathematically correct, and does indeed
    mean that DEBIET as a whole had a very small effect on the prediction.

    Parameters
    ----------
    df: dataframe
       The dataframe whose columns will be added together
    sep: str, optional (default='#')
       The seperator character. The group of a column is defined as everything before the first
       occurence of this character
    skipna: boolean, optional (default=True)
       Whether or not to ignore missing values in columns. If true, missing values are treated as
       0. If false, missing values are not ignored and the sum for that particular group/row
       combination will be missing as well.

    Returns
    -------
    summed_df: dataframe
        A dataframe with the same row-index as df, but with less columns. All the columns in the
        same group have been summed together.

    Examples
    --------
    >>> df = pd.DataFrame({
    >>>    'X#lag_0': [1, 2, 3],
    >>>    'X#lag_1': [1, 2, 3],
    >>>    'Y': [5, 5, 5]
    >>> })
    >>> sum_grouped_columns(df)
        X	Y
    0	2	5
    1	4	5
    2	6	5

    >>> # In this example, we use the new shapley values to make a shapley visualization
    >>> shaps = explainer.shap_values(X)
    >>> summed_shaps = sum_grouped_columns(shaps)
    >>> # Shapley plots often use the original feature values, but there is no single value
    >>> # To describe an entire group, so we have to use empty strings instead.
    >>> empty_X = pd.DataFrame(np.full_like(summed_shaps, "", dtype=str),
    >>>                        columns=summed_shaps.columns)
    >>> # make a force plot to explain the first instance
    >>> shap.force_plot(explainer.expected_value, summed_shaps.values[0,:], empty_X.iloc[0,:])
    """
    logger.debug("Now running sum_grouped_columns with sep={}, skipna={}".format(sep, skipna))
    foo = df.copy()
    groups = foo.columns.str.extract(r'^([^{}]*)'.format(sep)).values.reshape(-1)
    if skipna:
        result = foo.groupby(groups, axis=1).sum()
    else:
        result = foo.groupby(groups, axis=1).apply(lambda x: x.sum(skipna=False, axis=1))
    log_new_columns(result, df)
    return result
