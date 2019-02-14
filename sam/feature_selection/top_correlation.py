import pandas as pd
from sam.logging import log_dataframe_characteristics
import logging
logger = logging.getLogger(__name__)


def retrieve_top_n_correlations(df, goal_feature, n=5, grouped=True):
    """ Given a dataset, retrieve the top n absolute correlating features per group or in general

        Parameters
        ----------
        df: pandas dataframe
            Dataframe containing the features that have to be correlated.

        goal_feature: string
            Feature that is used to compare the correlation with other features

        n: int (default: 5)
            Number of correlating features that are returned

        grouped: boolean (default: true)
            Whether to group the features and take the top n of a group,
            or just the top n correlating features in general.
            Groups are created based on column name and the first item when
            column name is split by '_'.
            DEBIET#TOTAAL_lag_0 is in group DEBIET#TOTAAL

        Returns
        -------
        df: pandas dataframe
            A dataframe containing 3 columns(GROUP, index, goal_variable).
            Index contains the correlating features and
            goal_variable the correlation value.


        Examples
        --------
        >>> import pandas as pd
        >>> from sam.feature_engineering.rolling_features import BuildRollingFeatures
        >>> from sam.feature_selection.top_correlation import retrieve_top_n_correlations
        >>> import numpy as np
        >>> goal_feature = 'DEBIET#TOTAAL_lag_0'
        >>> df = pd.DataFrame({
        >>>                'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                'DEBIET#A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
        >>>                'DEBIET#B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
        >>> df['DEBIET#TOTAAL'] = df['DEBIET#A'] + df['DEBIET#B']
        >>> RollingFeatures = BuildRollingFeatures(rolling_type='lag', \\
        >>>     window_size = np.arange(12), lookback=0, keep_original=False)
        >>> res = RollingFeatures.fit_transform(df)
        >>> retrieve_top_n_correlations(res, goal_feature, n=2, grouped=True)
                   GROUP                index  DEBIET#TOTAAL_lag_0
        0       DEBIET#A       DEBIET#A_lag_0             0.838591
        1       DEBIET#A       DEBIET#A_lag_5             0.667537
        2       DEBIET#B       DEBIET#B_lag_0             0.897340
        3       DEBIET#B       DEBIET#B_lag_9             0.755929
        4  DEBIET#TOTAAL  DEBIET#TOTAAL_lag_9             0.944911
        5  DEBIET#TOTAAL  DEBIET#TOTAAL_lag_4             0.636884
        6           RAIN           RAIN_lag_9             0.944911
        7           RAIN           RAIN_lag_8             0.871695


        >>> retrieve_top_n_correlations(res, goal_feature, n=2, grouped=False)
                          index  DEBIET#TOTAAL_lag_0          GROUP
        39  DEBIET#TOTAAL_lag_9             0.944911  DEBIET#TOTAAL
        36           RAIN_lag_9             0.944911           RAIN


    """

    assert (goal_feature in df.columns), "Goal feature not found in columns!"

    logging.debug("Retrieving top n variables with goal variable {}, n={}, grouped={}".
                  format(goal_feature, n, grouped))

    pos_corr = df.corr().abs()  # get all positive correlations
    pos_corr = pos_corr.loc[goal_feature].reset_index()
    pos_corr = pos_corr.loc[pos_corr['index'] != goal_feature]
    pos_corr['GROUP'] = pos_corr['index'].apply(lambda x: x.split('_')[0])

    if grouped:
        pos_corr = \
            pos_corr.groupby('GROUP') \
            .apply(lambda x: x.nlargest(n, goal_feature))[
                ['index', goal_feature]].reset_index(drop=False)
        pos_corr = pos_corr.drop('level_1', axis=1)

    else:
        pos_corr = pos_corr.sort_values(goal_feature, ascending=False).head(n)

    corrs = df.corr()  # replace correlations with the correct negative ones
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = pos_corr.drop(goal_feature, axis=1).merge(corrs, on='index',
                                                      how='left')
    logging.info("Output from retrieve_top_n_correlations:")
    log_dataframe_characteristics(corrs, logging.INFO)
    return corrs


def retrieve_top_score_correlations(df, goal_feature, score=0.5):
    """ Function that returns the features that have a correlation
        above a certain threshold with the defined goal feature

        Parameters
        ----------
        df: pandas dataframe
            Dataframe containing the features that have to be correlated.

        goal_feature: string
            Feature that is used to compare the correlation with other features

        score: float (default: 0.5)
            absolute minimal correlation value

        Returns
        -------
        corrs: pandas dataframe
            A dataframe containing 2 columns(index, goal feature).
            Index contains the correlating features and
            goal feature the correlation values.

        Examples
        --------
        >>> import pandas as pd
        >>> from sam.feature_engineering.rolling_features import BuildRollingFeatures
        >>> from sam.feature_selection.top_correlation import retrieve_top_score_correlations
        >>> import numpy as np
        >>> goal_feature = 'DEBIET#TOTAAL_lag_0'
        >>> df = pd.DataFrame({
        >>>                'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        >>>                'DEBIET#A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
        >>>                'DEBIET#B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
        >>> df['DEBIET#TOTAAL'] = df['DEBIET#A'] + df['DEBIET#B']
        >>> RollingFeatures = BuildRollingFeatures(rolling_type='lag', \\
        >>>     window_size = np.arange(10), lookback=0, keep_original=False)
        >>> res = RollingFeatures.fit_transform(df)
        >>> retrieve_top_score_correlations(res, goal_feature, score=0.8)
                    index 	                DEBIET#TOTAAL_lag_0
                0 	DEBIET#TOTAAL_lag_9 	0.944911
                1 	RAIN_lag_9 	            -0.944911
                2 	DEBIET#B_lag_0 	        0.897340
                3 	RAIN_lag_8 	            0.871695
                4 	DEBIET#A_lag_0 	        0.838591

    """

    assert (goal_feature in df.columns), "Goal feature not found in columns!"
    logging.debug("Retrieving top n variables with goal variable {}, score={}".
                  format(goal_feature, score))

    pos_corr = df.corr().abs()  # get all positive correlations
    pos_corr = pos_corr.loc[goal_feature].reset_index()
    pos_corr = pos_corr.loc[
        (pos_corr['index'] != goal_feature) &
        (pos_corr[goal_feature] >= score)]
    pos_corr = pos_corr.sort_values(goal_feature, ascending=False)

    corrs = df.corr()  # replace correlations with the correct negative ones
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = pos_corr.drop(goal_feature, axis=1).merge(corrs, on='index',
                                                      how='left')
    logging.info("Output from retrieve_top_score_correlations:")
    log_dataframe_characteristics(corrs, logging.INFO)
    return corrs
