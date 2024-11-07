from typing import List

import numpy as np
import pandas as pd

COLORS = ["#2453bd", "#7497e3", "#c3d0eb", "#d1d8e8"]


def sam_quantile_plot(
    y_true: pd.Series,
    y_hat: pd.DataFrame,
    title: str = None,
    y_title: str = "",
    data_prop: int = None,
    y_range: int = None,
    date_range: list = None,
    colors: list = COLORS,
    outlier_min_q: int = None,
    predict_ahead: int = 0,
    res: str = None,
    interactive: bool = False,
    outliers: np.ndarray = None,
    outlier_window: int = 1,
    outlier_limit: int = 1,
    ignore_value: float = None,
    benchmark: pd.DataFrame = None,
    benchmark_color: str = "purple",
):
    """
    Uses the output from MLPTimeseriesRegressor predict function to create a quantile prediction
    plot. It plots the actual data, the prediction, and the quantiles as shaded regions.
    The plot displays a single prediction per timepoint (e.g. made 10 timepoints ago).
    The plot can be made for any predict_ahead, and can be resampled to any time-resolution.
    The plot also highlights outliers as all true values that fall outside the `outlier_min_q`th
    quantile.

    Note: this function only works when an even number of quantiles was used in the fit procedure!

    Parameters
    ---------
    y_true: pd.Series
        Pandas Series containing the actual values. Should have same index as y_hat.
    y_hat: pd.DataFrame
        Dataframe returned by the MLPTimeseriesRegressor .predict() function.
        Columns should contain at least `predict_lead_x_mean`, where x is predict ahead
        and for each quantile: `predict_lead_x_q_y` where x is the predict_ahead, and
        y is the quantile. So e.g.:
        `['predict_lead_0_q_0.25, predict_lead_0_q_0.75, predict_lead_mean']`
    title: string (default=None)
        Title for the plot.
    y_title: string (default='')
        Title to put along the yaxis (ylabel).
    data_prop: int (default=None)
        Proportion of data range to include outside true data maxima for plot range,
        if not set to None. This is only applied if y_range is None.
    y_range: list or None (default=None)
        The minimum and maximum for the y-axis if not set to None.
    date_range: list (default=None)
        The minimum and maximum date for the x-axis if not set to None.
        e.g.:['2019-11-13', '2019-11-26']
    colors: list of strings (default=['#2453bd', '#7497e3', '#c3d0eb'])
        should be a valid colorstring for each quantile (first in list is narrowest quantile etc.)
    outlier_min_q: int (default=None)
        Outlier number to use for determining 'invalid' samples.
        1 indicates the narrowest CI.
        If 3 quantiles are used in the fit procedure, this can be either 1, 2, or 3.
        In this situation, if outlier_min_q is set to 3, all true values that fall outside of the
        third quantile are highlighted in the plot.
        Cannot be used in conjunction with `outliers`.
    predict_ahead: int (default=0)
        Number of samples ago that prediction was made.
        Should be one that was included in the MLPTimeseriesRegressor fit procedure.
    res: string (default=None)
        Time resolution to resample data to. Should be interpretable by pandas resamples
        (e.g. '5min', '1D' etc.). For this to work, the data must have datetime indices.
    interactive: bool (default=False)
        Returns a matplotlib figure if False, otherwise returns a plotly figure.
    outliers: array-like (default=None)
        Alternatively to the outlier_min_q argument, you can pass a boolean array of outliers here.
        This allows the user to specify their own rules that determine whether a sample is an
        outlier or not. Should be same length as y_hat, and should either have same index,
        or no index at all (np array) or list.
        Cannot be used in conjunction with `outlier_min_q`.
    outlier_window: int (default=1)
        the window size in which at least `outlier_limit` should be outside of `outlier_min_q`
    outlier_limit: int (default=1)
        the minimum number of outliers within outlier_window to be outside of `outlier_min_q`
    ignore_value: float (default=None)
        value to ignore during resampling (e.g. 0 for pumps that often go off)
    benchmark: pd.DataFrame
        The benchmark used to determine R2 of y_hat, for example a dataframe returned by the
        MLPTimeseriesRegressor.predict() function. Columns should contain at least
        `predict_lead_x_mean`, where x is predict ahead and for each quantile: `predict_lead_x_q_y`
        where x is the predict_ahead, and y is the quantile. So e.g.:
        `['predict_lead_0_q_0.25, predict_lead_0_q_0.75, predict_lead_mean']`
    benchmark_color: string (default='purple')
        a valid colorstring for the benchmark line/scatter color

    Returns
    ------
    fig: matplotlib.pyplot.figure if interactive=False else go.Figure
    """

    if outlier_min_q is not None and outliers is not None:
        raise ValueError("outlier_min_q and outliers cannot be used simultaneously")

    if ignore_value is not None and res is None:
        raise ValueError(
            "ignore value should only be set when using resampling (res should not be None)"
        )

    if (y_title == "") and y_true.name:
        y_title = y_true.name

    # copy to make sure we don't modify the original
    y_true = y_true.copy()
    y_hat = y_hat.copy()

    # shift prediction back to match y_true
    y_hat = y_hat.shift(predict_ahead)

    # apply date range to data to speed up the rest
    if date_range is None:
        start, end = y_true.index.min(), y_true.index.max()
    else:
        start, end = date_range[0], date_range[1]
    y_true = y_true.loc[start:end]
    y_hat = y_hat.loc[start:end]

    # set ignore_values to nan so they are ignored in the resampling
    if ignore_value is None:
        ignore_timepoints = y_true.apply(lambda x: False)
    else:
        ignore_timepoints = y_true == ignore_value

    y_true.loc[ignore_timepoints] = np.nan
    y_hat.loc[ignore_timepoints] = np.nan

    # same pre-processing steps for the benchmark
    if benchmark is not None:
        benchmark = benchmark.copy()
        benchmark = benchmark.shift(predict_ahead)
        benchmark = benchmark.loc[start:end]
        benchmark.loc[ignore_timepoints] = np.nan

    # resample to desired resolution
    if res is not None:
        y_true = y_true.resample(res).mean()
        y_hat = y_hat.resample(res).mean()
        if benchmark is not None:
            benchmark = benchmark.resample(res).mean()

    # create figure
    if interactive:
        _plot_func = _interactive_quantile_plot
    else:
        _plot_func = _static_quantile_plot

    fig = _plot_func(
        y_true=y_true,
        y_hat=y_hat,
        title=title,
        y_title=y_title,
        data_prop=data_prop,
        y_range=y_range,
        date_range=date_range,
        colors=colors,
        outlier_min_q=outlier_min_q,
        predict_ahead=predict_ahead,
        outliers=outliers,
        outlier_window=outlier_window,
        outlier_limit=outlier_limit,
        benchmark=benchmark,
        benchmark_color=benchmark_color,
    )

    return fig


def _interactive_quantile_plot(
    y_true: pd.Series,
    y_hat: pd.DataFrame,
    title: str = None,
    y_title: str = "",
    data_prop: int = None,
    y_range: int = None,
    date_range: list = None,
    colors: list = COLORS,
    outlier_min_q: int = None,
    predict_ahead: int = 0,
    outliers: np.ndarray = None,
    outlier_window: int = 1,
    outlier_limit: int = 1,
    benchmark: pd.DataFrame = None,
    benchmark_color: str = "purple",
):
    import plotly.graph_objs as go

    # some bookkeeping before we start plotting
    these_cols = [c for c in y_hat.columns if "predict_lead_%d_q_" % predict_ahead in c]

    # figure out how to sort the columns
    col_order = np.argsort([float(c.split("_")[-1]) for c in these_cols])

    # determine number of quantiles
    n_quants = int((len(these_cols)) / 2)

    # setup plotly figure
    fig = go.Figure()

    # loop over quantiles to create shaded regions
    for i in list(range(0, n_quants))[::-1]:
        highcol = these_cols[col_order[n_quants + i]]
        lowcol = these_cols[col_order[n_quants - 1 - i]]
        this_ci = float(highcol.split("_")[-1]) - float(lowcol.split("_")[-1])

        y_hat_dropna = y_hat.dropna()

        fig.add_trace(
            go.Scatter(
                x=y_hat_dropna.index,
                y=y_hat_dropna[highcol],
                fill=None,
                mode="lines",
                line={"width": 0.1},
                line_color=colors[i],
                name="%.5f CI" % this_ci,
                showlegend=False,
                legendgroup="%.5f CI" % this_ci,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=y_hat_dropna.index,
                y=y_hat_dropna[lowcol],
                fill="tonexty",
                mode="lines",
                line={"width": 0.1},
                line_color=colors[i],
                name="%.5f CI" % this_ci,
                legendgroup="%.5f CI" % this_ci,
            )
        )

    # now draw the mean prediction and actuals
    fig.add_trace(
        go.Scatter(
            x=y_hat.index,
            y=y_hat["predict_lead_%d_mean" % predict_ahead],
            line_color=colors[0],
            name="predicted",
        )
    )

    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark["predict_lead_%d_mean" % predict_ahead],
                line_color=benchmark_color,
                name="benchmark",
            )
        )

    fig.add_trace(go.Scatter(x=y_true.index, y=y_true, line_color="black", name="true"))

    # now plot outliers
    if outlier_min_q is not None or outliers is not None:
        # if outliers is not passed, determine them through use of outlier_min_q
        if outlier_min_q is not None and outliers is None:
            valid_low = y_hat[these_cols[col_order[n_quants - 1 - (outlier_min_q - 1)]]]
            valid_high = y_hat[these_cols[col_order[n_quants + (outlier_min_q - 1)]]]
            outliers = (y_true >= valid_high) | (y_true <= valid_low)
            outliers = outliers.astype(int)
            k = np.ones(outlier_window)
            outliers = (
                np.convolve(outliers, k, mode="full")[: len(outliers)] >= outlier_limit
            ).astype(bool)

        fig.add_trace(
            go.Scatter(
                x=y_true[outliers].index,
                y=y_true[outliers],
                mode="markers",
                marker={"color": "red"},
                name="outlier",
            )
        )

    # set some plot properties
    if data_prop is not None and y_range is None:
        y_range = _get_y_range(y_true, data_prop)

    if title is not None:
        fig.layout.update(title=title)
    if y_range is not None:
        fig.layout.update(yaxis_range=y_range)
    if date_range is not None:
        fig.layout.update(xaxis_range=date_range)

    fig.layout.update(
        yaxis_title=y_title,
        width=1000,
        height=500,
    )

    return fig


def _static_quantile_plot(
    y_true: pd.Series,
    y_hat: pd.DataFrame,
    title: str = None,
    y_title: str = "",
    data_prop: int = None,
    y_range: int = None,
    date_range: list = None,
    colors: list = COLORS,
    outlier_min_q: int = None,
    predict_ahead: int = 0,
    outliers: np.ndarray = None,
    outlier_window: int = 1,
    outlier_limit: int = 1,
    benchmark: pd.DataFrame = None,
    benchmark_color: str = "purple",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # some bookkeeping before we start plotting
    these_cols = [c for c in y_hat.columns if "predict_lead_%d_q_" % predict_ahead in c]

    # figure out how to sort the columns
    col_order = np.argsort([float(c.split("_")[-1]) for c in these_cols])

    # determine number of quantiles
    n_quants = int((len(these_cols)) / 2)

    # setup plotly figure
    fig = plt.figure(figsize=(15, 7.5))

    # loop over quantiles to create shaded regions
    for i in list(range(0, n_quants))[::-1]:
        highcol = these_cols[col_order[n_quants + i]]
        lowcol = these_cols[col_order[n_quants - 1 - i]]
        this_ci = float(highcol.split("_")[-1]) - float(lowcol.split("_")[-1])

        plt.fill_between(
            y_hat.index,
            y_hat[lowcol],
            y_hat[highcol],
            alpha=0.25,
            color=colors[i],
            label="%.5f CI" % this_ci,
        )

    # now draw the mean prediction and actuals
    plt.plot(
        y_hat.index,
        y_hat["predict_lead_%d_mean" % predict_ahead],
        color=colors[0],
        label="predicted",
    )

    if benchmark is not None:
        plt.plot(
            benchmark.index,
            benchmark["predict_lead_%d_mean" % predict_ahead],
            color=benchmark_color,
            label="benchmark",
        )

    plt.plot(y_true.index, y_true, color="black", label="true")

    # now plot outliers
    if outlier_min_q is not None or outliers is not None:
        # if outliers is not passed, determine them through use of outlier_min_q
        if outlier_min_q is not None and outliers is None:
            valid_low = y_hat[these_cols[col_order[n_quants - 1 - (outlier_min_q - 1)]]]
            valid_high = y_hat[these_cols[col_order[n_quants + (outlier_min_q - 1)]]]
            outliers = (y_true >= valid_high) | (y_true <= valid_low)
            outliers = outliers.astype(int)
            k = np.ones(outlier_window)
            outliers = (
                np.convolve(outliers, k, mode="full")[: len(outliers)] >= outlier_limit
            ).astype(bool)

        plt.plot(
            y_true[outliers].index,
            y_true[outliers],
            "o",
            ms=5,
            color="r",
            label="outlier",
        )

    # set some plot properties
    if data_prop is not None and y_range is None:
        y_range = _get_y_range(y_true, data_prop)

    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    plt.ylabel(y_title)
    sns.despine(offset=10)
    if y_range is not None:
        plt.ylim(y_range)
    if date_range is not None:
        plt.xlim(pd.to_datetime(date_range))

    return fig


def _get_y_range(y: pd.DataFrame, data_prop: int) -> List[float]:
    """Calculates the proportional range of `y` given `data_prop`

    Parameters
    ----------
    y : pd.DataFrame
        Pandas DataFrame with single row
    data_prop: int
        Proportion of data range to include outside data maxima for plot range

    Returns
    -------
    List[float]
        List of length 2, with first index the minimum and second index the maximum of the
        proportional range of `y`
    """
    data_range = y.max() - y.min()
    ymax = y.max() + data_range * data_prop
    ymin = y.min() - data_range * data_prop
    y_range = [ymin, ymax]

    return y_range
