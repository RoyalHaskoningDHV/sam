import pandas as pd


def create_lag_correlation(df, goal_variable):
    """ Creates a new dataframe that contains the correlation of a goal_variable
        with other variables in the dataframe based on the output from
        BuildRollingFeatures. The results are processed for easy visualization,
        with a column for the lag and then correlation per feature.

        Parameters
        ----------
        df: pandas dataframe
            input dataframe contains the lag correlation variable
            and other variable

        goal_variable: string
            Goal variable of which the correlations have to be compared against.

        Returns
        -------
        tab: pandas dataframe
            A dataframe with the correlations and shift.
            The column header contains the feature name.

        Examples
        --------
        >>> import pandas as pd
        >>> from sam.feature_engineering.rolling_features import BuildRollingFeatures
        >>> import numpy as np
        >>> goal_feature = 'DEBIET#TOTAAL_lag_0'
        >>> df = pd.DataFrame({
                           'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           'DEBIET#A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
                           'DEBIET#B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
        >>> df['DEBIET#TOTAAL'] = df['DEBIET#A'] + df['DEBIET#B']
        >>> RollingFeatures = BuildRollingFeatures(rolling_type='lag', \
                window_size = np.arange(12), lookback=0, keep_original=False)
        >>> res = RollingFeatures.fit_transform(df)
        >>> test = create_lag_correlation(res, goal_feature)
        >>> test
            LAG  Debiet#A  Debiet#B  Debiet#totaal      Rain
        0     0  0.838591  0.897340       1.000000 -0.017557
        1     1  0.436484  0.102808       0.292156  0.204983
        2     2  0.287863 -0.401768      -0.080807  0.672316
        3     3 -0.388095 -0.140876      -0.294663  0.188438
        4     4 -0.632980 -0.509307      -0.636884 -0.227071
        5     5 -0.667537 -0.367268      -0.575000 -0.048162
        6     6 -0.152832  0.615239       0.264925  0.110876
        7     7  0.457496 -0.107833       0.302326 -0.719702
        8     8  0.291111  0.039253       0.242065  0.871695
        9     9  0.188982  0.755929       0.944911 -0.944911
        10   10  1.000000 -1.000000      -1.000000  1.000000

    """
    corr_table = df.corr()[goal_variable].reset_index()
    corr_table['LAG'] = corr_table['index'].apply(lambda x: x.split('_')[-1])
    corr_table['GROUP'] = corr_table['index'].apply(lambda x: x.split('_')[0])
    tab = pd.pivot_table(corr_table, values=goal_variable, index='LAG',
                         columns='GROUP')
    tab = tab.reset_index()
    tab['LAG'] = pd.to_numeric(tab['LAG'])
    tab = tab.sort_values('LAG')
    tab.columns.name = ""
    tab = tab.reset_index(drop=True)
    return tab
