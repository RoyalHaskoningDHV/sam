def plot_lag_correlation(df, ylim=None):
    ''' Visualize the correlations for rolling features

        Parameters
        ----------
        df: pandas dataframe (columns: LAG and columns to visualise)
            Dataframe containing the data that has to be visualized

        ylim : list[min, max]
            In order to customize the y-axis limit

        Returns
        -------
        ax: seaborn plot
            Plot containing the visualization of the data

        Examples
        --------
        >>> import pandas as pd
        >>> from sam.feature_engineering.rolling_features import BuildRollingFeatures
        >>> from sam.feature_selection.lag_correlation import create_lag_correlation
        >>> from sam.visualization.rolling_correlations import plot_lag_correlation
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
        >>> test = create_lag_correlation(res, goal_feature)
        >>> plot_lag_correlation(test)
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for col in df.columns:
        if col == 'LAG':
            continue
        sns.lineplot(x='LAG', y=col, data=df, label=col, ax=ax)

    if ylim:
        ax.set_ylim(ylim)

    ax.set_ylabel('Correlation')
    ax.legend(loc='center right',
              bbox_to_anchor=(0.35, 0.5, 1, 0))

    return ax
