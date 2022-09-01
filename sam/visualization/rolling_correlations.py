import numpy as np
import pandas as pd


def plot_lag_correlation(df: pd.DataFrame, ylim_min: int = None, ylim_max: int = None):
    """
    Visualize the correlations for rolling features

    Parameters
    ----------
    df: pandas dataframe (columns: LAG and columns to visualise)
        Dataframe containing the data that has to be visualized.

    ylim_min: int (default = None)
        minimun for customizing the y-axis limit.
    ylim_max: int (default = None)
        maximum for customizing the y-axis limit.

    Returns
    -------
    ax: seaborn plot
        Plot containing the visualization of the data

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> from sam.exploration import lag_correlation
    >>> from sam.visualization.rolling_correlations import plot_lag_correlation
    >>> import numpy as np
    >>> goal_feature = 'DEBIET#TOTAAL'
    >>> df = pd.DataFrame({
    ...                'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0,
    ...                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...                'DEBIET#A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
    ...                'DEBIET#B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
    >>> df['DEBIET#TOTAAL'] = df['DEBIET#A'] + df['DEBIET#B']
    >>> test = lag_correlation(df, goal_feature)
    >>> f = plot_lag_correlation(test)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _, ax = plt.subplots(figsize=(15, 10))
    plt.axhline(0, ls="--", c="k")

    df_long = df.melt(id_vars="LAG")
    sns.lineplot(
        data=df_long,
        x="LAG",
        hue="variable",
        y="value",
        ax=ax,
        ls="--",
        marker="o",
        ms=12,
    )

    if ylim_min and ylim_max:
        ax.set_ylim(bottom=ylim_min, top=ylim_max)

    for col in df.drop("LAG", axis=1).columns:
        maxi = np.argmax(np.abs(df[col].values))
        plt.plot(df["LAG"].iloc[maxi], df[col].iloc[maxi], marker="o", ms=8, c="w")

    ax.set_ylabel("Correlation")
    ax.legend(loc="center right", bbox_to_anchor=(0.35, 0.5, 1, 0))
    sns.despine()
    plt.tight_layout()

    return ax
