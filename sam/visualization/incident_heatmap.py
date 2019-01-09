import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def make_incident_heatmap(df, resolution='row', row_column='id', value_column='incident',
                          time_column=None, normalize=False, figsize=(24, 4), **kwargs):
    """
    Create and return a heatmap for incident occurence. This can be used to visualize e.g.
    the count of outliers/threshold surpassings/warnings given over time.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    df : Pandas DataFrame
        Contains the data to be plotted in long format
    resolution : string (default="row")
        The aggregation level to be plotted. Is either "row" when every
        row needs to be plotted, or e.g. H, D, W, M for aggregations over time.
    row_column : string (default="id")
        The column that used to split the rows of the heatmap
    value_column : string (default="incident")
        The column containing the count to be plotted in the heatmap
    time_column : string (default=None)
        The column containing the time for the x-axis. When left None, the
        index will be used. Aggregation based on the resolution parameter
        is done on this column.
    normalize : boolean (default=False)
        Normalize the aggregated values
    figsize : tuple of floats (default=(24,4))
        Size of the output figure, must be set before initialization.
    **kwargs : any arguments to pass to sns.heatmap()

    Returns
    -------
    plot:  matplotlib.axes._subplots.AxesSubplot object
        a plot containing the heatmap. Can be edited further,
        or printed to the output.

    Examples
    --------
    >>> # Initialize a random dataframe
    >>> rng = pd.date_range('1/1/2011', periods=150, freq='D')
    >>> ts = pd.DataFrame({'values': np.random.randn(len(rng)),
                           'id': np.random.choice(['A','B','C'], len(rng))},
                           index=rng, columns=['values','id'])
    >>>
    >>> # Create some incidents
    >>> ts['incident'] = 0
    >>> ts.loc[ts['values'] > .5, 'incident'] = 1
    >>>
    >>> # Create the heatmap
    >>> ax = incident_heatmap(ts, resolution='W', annot=True, cmap='Reds')
    """
    df = df.copy()

    # Resample the data if needed
    if resolution != 'row':
        df_grouped = df.groupby([row_column, pd.Grouper(key=time_column, freq=resolution)])
        df_grouped = df_grouped[value_column].sum().unstack(fill_value=0)
    else:
        if time_column is not None:
            # Use a specific column instead of the index
            df_grouped = df.pivot(row_column, time_column, value_column).fillna(0)
        else:
            # Use the index, so first reset_index
            df_grouped = df.reset_index().pivot(row_column, 'index', value_column).fillna(0)

    if normalize:
        df_grouped = df_grouped / df_grouped.values.max()

    # We typically want a wide figure
    plt.rcParams['figure.figsize'] = figsize

    # Initialize heatmap
    ax = sns.heatmap(df_grouped, **kwargs)

    return ax
