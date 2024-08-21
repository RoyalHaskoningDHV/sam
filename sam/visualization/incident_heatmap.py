from math import ceil, floor
from typing import Iterable, Tuple

import pandas as pd


def plot_incident_heatmap(
    df: pd.DataFrame,
    resolution: str = "row",
    row_column: str = "id",
    value_column: str = "incident",
    time_column: str = None,
    normalize: bool = False,
    figsize: Iterable[Tuple[int, int]] = (24, 4),
    xlabel_rotation: int = 30,
    datefmt: str = None,
    **kwargs,
):
    """
    Create and return a heatmap for incident occurence. This can be used to visualize e.g.
    the count of outliers/threshold surpassings/warnings given over time.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    df: Pandas DataFrame
        Contains the data to be plotted in long format
    resolution: string (default="row")
        The aggregation level to be plotted. Is either "row" when every
        row needs to be plotted, or e.g. H, D, W, M for aggregations over time.
    row_column: string (default="id")
        The column that used to split the rows of the heatmap
    value_column: string (default="incident")
        The column containing the count to be plotted in the heatmap
    time_column: string (default=None)
        The column containing the time for the x-axis. When left None, the
        index will be used. Aggregation based on the resolution parameter
        is done on this column.
    normalize: boolean (default=False)
        Normalize the aggregated values
    figsize: tuple of floats (default=(24,4))
        Size of the output figure, must be set before initialization.
    xlabel_rotation: numeric, optional (default=30)
        The rotation of the x-axis date labels. Rotation is counterclockwise, beginning
        with the text lying horizontally. By default, rotate 30 degrees.
    datefmt: string, optional (default=None)
        Optionally, the format of the x-axis date labels. By default, use
        `%Y-%m-%dT%H:%M:%S%f`
    **kwargs:
        any arguments to pass to sns.heatmap()

    Returns
    -------
    plot:  matplotlib.axes._subplots.AxesSubplot object
        a plot containing the heatmap. Can be edited further,
        or printed to the output.

    Examples
    --------
    >>> from sam.visualization import plot_incident_heatmap
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Initialize a random dataframe
    >>> rng = pd.date_range('1/1/2011', periods=150, freq='D')
    >>> ts = pd.DataFrame({'values': np.random.randn(len(rng)),
    ...                    'id': np.random.choice(['A','B','C'], len(rng))},
    ...                     index=rng, columns=['values','id'])
    >>>
    >>> # Create some incidents
    >>> ts['incident'] = 0
    >>> ts.loc[ts['values'] > .5, 'incident'] = 1
    >>>
    >>> # Create the heatmap
    >>> fig = plot_incident_heatmap(
    ...     ts, resolution='W', annot=True, cmap='Reds', datefmt="%Y, week %W"
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()

    # Resample the data if needed
    if resolution != "row":
        df_grouped = df.groupby([row_column, pd.Grouper(key=time_column, freq=resolution)])
        df_grouped = df_grouped[value_column].sum().unstack(fill_value=0)
    else:
        if time_column is not None:
            # Use a specific column instead of the index
            df_grouped = df.pivot(
                index=row_column, columns=time_column, values=value_column
            ).fillna(0)
        else:
            # Use the index, so first reset_index
            df_grouped = (
                df.reset_index()
                .pivot(index=row_column, columns="index", values=value_column)
                .fillna(0)
            )

    if normalize:
        df_grouped = df_grouped / df_grouped.values.max()

    # We typically want a wide figure
    plt.figure(figsize=figsize)

    # Initialize heatmap
    ax = sns.heatmap(df_grouped, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_rotation)

    # Set y limits. On some versions/platforms of seaborn, these are set at 0.5
    # instead of 0, so we round up the higher limit, and round down the lower limit.
    # If the limits are correctly set as integers, these lines have no effect.
    # It is totally possible for ax.get_ylim() to return (high, low), in which
    # case we need to switch them with `sorted` to obtain (low, high) instead.
    y_low, y_high = sorted(ax.get_ylim())
    ax.set_ylim(floor(y_low), ceil(y_high))

    # If desired, add custom date format to the x-axis
    if datefmt:
        ax.set_xticklabels(df_grouped.columns.strftime(datefmt))

    return ax
