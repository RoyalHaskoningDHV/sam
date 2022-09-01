import logging

import pandas as pd
from sam.logging_functions import log_dataframe_characteristics

logger = logging.getLogger(__name__)


def top_n_correlations(
    df: pd.DataFrame,
    goal_feature: str,
    n: int = 5,
    grouped: bool = True,
    sep: str = "#",
):
    """
    Given a dataset, retrieve the top n absolute correlating features per group or in general

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the features that have to be correlated.
    goal_feature: str
        Feature that is used to compare the correlation with other features
    n: int (default: 5)
        Number of correlating features that are returned
    grouped: bool (default: True)
        Whether to group the features and take the top n of a group, or just the top n
        correlating features in general. Groups are created based on column name, and
        are all characters before the first occurence of the sep. s
        For example, if the sep is '#', then DEBIET_TOTAAL#lag_0 is in group DEBIET_TOTAAL
    sep: str (default: '#')
        The seperator character. The group of a column is defined as everything before the first
        occurence of this character. Only relevant if grouped is True

    Returns
    -------
    df: pd.DataFrame
        If grouped is true, a dataframe containing 3 columns (GROUP, index, goal_variable) is
        returned, else a dataframe containing 2 columns (index, goal_variable) is returned.
        index contains the correlating features and goal_variable the correlation value.
        GROUP contains the group.


    Examples
    --------
    >>> import pandas as pd
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> from sam.exploration import top_n_correlations
    >>> import numpy as np
    >>> goal_feature = 'DEBIET_TOTAAL#lag_0'
    >>> df = pd.DataFrame({
    ...                'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...                'DEBIET_A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
    ...                'DEBIET_B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
    >>> df['DEBIET_TOTAAL'] = df['DEBIET_A'] + df['DEBIET_B']
    >>> RollingFeatures = BuildRollingFeatures(rolling_type='lag',
    ...     window_size = np.arange(12), lookback=0, keep_original=False)
    >>> res = RollingFeatures.fit_transform(df)
    >>> top_n_correlations(res, goal_feature, n=2, grouped=True, sep='#')
               GROUP                 index  DEBIET_TOTAAL#lag_0
    0       DEBIET_A       DEBIET_A#lag_10             1.000000
    1       DEBIET_A        DEBIET_A#lag_0             0.838591
    2       DEBIET_B       DEBIET_B#lag_10            -1.000000
    3       DEBIET_B        DEBIET_B#lag_0             0.897340
    4  DEBIET_TOTAAL  DEBIET_TOTAAL#lag_10            -1.000000
    5  DEBIET_TOTAAL   DEBIET_TOTAAL#lag_9             0.944911
    6           RAIN           RAIN#lag_10             1.000000
    7           RAIN            RAIN#lag_9            -0.944911

    >>> top_n_correlations(res, goal_feature, n=2, grouped=False)
                      index  DEBIET_TOTAAL#lag_0
    0  DEBIET_TOTAAL#lag_10                 -1.0
    1       DEBIET_A#lag_10                  1.0
    """

    if goal_feature not in df.columns:
        raise ValueError("Goal feature not found in columns!")

    logging.debug(
        "Retrieving top n variables with goal variable {}, n={}, grouped={}".format(
            goal_feature, n, grouped
        )
    )

    pos_corr = df.corr().abs()  # get all positive correlations
    pos_corr = pos_corr.loc[goal_feature].reset_index()
    pos_corr = pos_corr.loc[pos_corr["index"] != goal_feature]

    if grouped:
        pos_corr["GROUP"] = pos_corr["index"].apply(lambda x: x.split(sep)[0])
        pos_corr = (
            pos_corr.groupby("GROUP")
            .apply(lambda x: x.nlargest(n, goal_feature))[["index", goal_feature]]
            .reset_index(drop=False)
        )
        pos_corr = pos_corr.drop("level_1", axis=1)

    else:
        pos_corr = pos_corr.sort_values(goal_feature, ascending=False).head(n)

    corrs = df.corr()  # replace correlations with the correct negative ones
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = pos_corr.drop(goal_feature, axis=1).merge(corrs, on="index", how="left")
    logging.info("Output from retrieve_top_n_correlations:")
    log_dataframe_characteristics(corrs, logging.INFO)
    return corrs


def top_score_correlations(df: pd.DataFrame, goal_feature: str, score: float = 0.5):
    """
    Returns the features that have a correlation
    above a certain threshold with the defined goal feature

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the features that have to be correlated.

    goal_feature: str
        Feature that is used to compare the correlation with other features

    score: float (default: 0.5)
        absolute minimal correlation value

    Returns
    -------
    corrs: pd.DataFrame
        A dataframe containing 2 columns(index, goal feature).
        Index contains the correlating features and
        goal feature the correlation values.

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> from sam.exploration import top_score_correlations
    >>> import numpy as np
    >>> goal_feature = 'DEBIET_TOTAAL#lag_0'
    >>> df = pd.DataFrame({
    ...                'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...                'DEBIET_A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
    ...                'DEBIET_B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
    >>> df['DEBIET_TOTAAL'] = df['DEBIET_A'] + df['DEBIET_B']
    >>> RollingFeatures = BuildRollingFeatures(rolling_type='lag', \\
    ...     window_size = np.arange(10), lookback=0, keep_original=False)
    >>> res = RollingFeatures.fit_transform(df)
    >>> top_score_correlations(res, goal_feature, score=0.8)
                     index  DEBIET_TOTAAL#lag_0
    0  DEBIET_TOTAAL#lag_9             0.944911
    1           RAIN#lag_9            -0.944911
    2       DEBIET_B#lag_0             0.897340
    3           RAIN#lag_8             0.871695
    4       DEBIET_A#lag_0             0.838591
    """

    if goal_feature not in df.columns:
        raise ValueError("Goal feature not found in columns!")
    logging.debug(
        "Retrieving top n variables with goal variable {}, score={}".format(goal_feature, score)
    )

    pos_corr = df.corr().abs()  # get all positive correlations
    pos_corr = pos_corr.loc[goal_feature].reset_index()
    pos_corr = pos_corr.loc[
        (pos_corr["index"] != goal_feature) & (pos_corr[goal_feature] >= score)
    ]
    pos_corr = pos_corr.sort_values(goal_feature, ascending=False)

    corrs = df.corr()  # replace correlations with the correct negative ones
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = pos_corr.drop(goal_feature, axis=1).merge(corrs, on="index", how="left")
    logging.info("Output from retrieve_top_score_correlations:")
    log_dataframe_characteristics(corrs, logging.INFO)
    return corrs
