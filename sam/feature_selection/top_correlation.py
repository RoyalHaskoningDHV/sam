import pandas as pd


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

    corrs = df.corr().abs()  # get all positive correlations
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = corrs.loc[corrs['index'] != goal_feature]
    corrs['GROUP'] = corrs['index'].apply(lambda x: x.split('_')[0])

    if grouped:
        corrs = \
          corrs.groupby('GROUP').apply(lambda x: x.nlargest(n, goal_feature))[
            ['index', goal_feature]].reset_index(drop=False)
        corrs = corrs.drop('level_1', axis=1)
        return corrs
    else:
        corrs = corrs.sort_values(goal_feature, ascending=False).head(n)
        return corrs


def retrieve_top_correlations(df, goal_feature, score=0.5):
    """ Function that returns the features that have a absolute correlation
        above a certain threshold with the defined goal feature

        Parameters
        ----------
        df: pandas dataframe
            Dataframe containing the features that have to be correlated.

        goal_feature: string
            Feature that is used to compare the correlation with other features

        perc: float (default: 0.5)
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
        >>> from sam.feature_selection.top_correlation import retrieve_top_perc_correlations
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
        >>> retrieve_top_perc_correlations(res, goal_feature, score=0.8)
                          index  Debiet#totaal_lag_0
        39  Debiet#totaal_lag_9             0.944911
        36           Rain_lag_9             0.944911
        2        Debiet#B_lag_0             0.897340
        32           Rain_lag_8             0.871695
        1        Debiet#A_lag_0             0.838591

    """

    assert (goal_feature in df.columns), "Goal feature not found in columns!"

    pos_corrs = df.corr().abs()  # get all positive correlations
    pos_corrs = pos_corrs.loc[goal_feature].reset_index()
    rel_cols = pos_corrs.loc[
        (pos_corrs['index'] != goal_feature) &
        (pos_corrs[goal_feature] >= score)][
        'index'].values
    corrs = df.corr()
    corrs = corrs.loc[goal_feature].reset_index()
    corrs = corrs[corrs['index'].isin(rel_cols.values)]
    corrs = corrs.sort_values(goal_feature, ascending=False)
    return corrs
