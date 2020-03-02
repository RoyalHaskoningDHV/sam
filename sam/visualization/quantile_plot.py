import matplotlib.pyplot as plt
import numpy as np


def sam_quantile_plot(
        y_true,
        y_hat,
        title=None,
        y_title='',
        data_prop=None,
        y_range=None,
        date_range=None,
        colors=['#2453bd', '#7497e3', '#c3d0eb'],
        outlier_min_q=None,
        predict_ahead=0,
        res=None,
        interactive=False,
        outliers=None):
    """
    Uses the output from SamQuantileMLPs predict function to create a quantile prediction plot.
    It plots the actual data, the prediction, and the quantiles as shaded regions.
    The plot displays a single prediction per timepoint (e.g. made 10 timepoints ago).
    The plot can be made for any predict_ahead, and can be resampled to any time-resolution.
    The plot also highlights outliers as all true values that fall outside the `outlier_min_q`th
    quantile.

    Note: this function only works when an even number of quantiles was used in the fit procedure!

    Parameters
    ---------
    y_true: pd.DataFrame
        Pandas DataFrame with single row (the actual values) should have same index as y_hat.
    y_hat: pd.DataFrame
        Dataframe returned by the SamQuantileMLP .predict() function.
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
        Should be one that was included in the SamQuantileMLP fit procedure.
    res: string (default=None)
        Time resolution to resample data to. Should be interpretable by pandas resamples
        (e.g. '5min', '1D' etc.). For this to work, the data must have datetime indices.
    plot_type: bool (default=False)
        Returns a matplotlib figure if False, otherwise returns a plotly figure/
        (Click here) <https://plot.ly/python/getting-started/>`_ to see how to install plotly
        and how to display figures inline in jupyter notebooks or lab.
    outliers: array-like (default=None)
        Alternatively to the outlier_min_q argument, you can pass a boolean array of outliers here.
        This allows the user to specify their own rules that determine whether a sample is an
        outlier or not. Should be same length as y_hat, and should either have same index,
        or no index at all (np array) or list.
        Cannot be used in conjunction with `outlier_min_q`.

    Returns
    ------
    fig: matplotlib figure
    """

    assert not (outlier_min_q is not None and outliers is not None),\
        'outlier_min_q and outliers cannot be used simultaneously'

    import seaborn as sns
    if interactive:
        import plotly.graph_objs as go
        from plotly.offline import plot

    # shift prediction back to match y_true
    y_hat = y_hat.shift(predict_ahead)

    # apply date range to data to speed up the rest
    if date_range is not None:
        y_true = y_true[date_range[0]:date_range[1]]
        y_hat = y_hat[date_range[0]:date_range[1]]

    # resample to desired resolution
    if res is not None:
        y_true = y_true.resample(res).mean()
        y_hat = y_hat.resample(res).mean()

    # some bookkeeping before we start plotting
    these_cols = [c for c in y_hat.columns if 'predict_lead_%d_q_' % predict_ahead in c]

    # figure out how to sort the columns
    col_order = np.argsort([float(c.split('_')[-1]) for c in these_cols])

    # determine number of quantiles
    n_quants = int((len(these_cols))/2)

    # setup plotly figure
    if interactive:
        fig = go.Figure()
    else:
        fig = plt.figure(figsize=(15, 7.5))

    # loop over quantiles to create shaded regions
    for i in list(range(0, n_quants))[::-1]:

        highcol = these_cols[col_order[n_quants+i]]
        lowcol = these_cols[col_order[n_quants-1-i]]
        this_ci = float(highcol.split('_')[-1]) - float(lowcol.split('_')[-1])

        if not interactive:

            plt.fill_between(
                y_hat.index,
                y_hat[lowcol],
                y_hat[highcol],
                alpha=0.25,
                color=colors[i],
                label='%.3f CI' % this_ci)

        else:

            y_hat_dropna = y_hat.dropna()

            fig.add_trace(go.Scatter(
                x=y_hat_dropna.index,
                y=y_hat_dropna[highcol],
                fill=None,
                mode='lines',
                line={'width': 0.1},
                line_color=colors[i],
                name='%.3f CI' % this_ci,
                showlegend=False,
                legendgroup='%.3f CI' % this_ci))

            fig.add_trace(go.Scatter(
                x=y_hat_dropna.index,
                y=y_hat_dropna[lowcol],
                fill='tonexty',
                mode='lines',
                line={'width': 0.1},
                line_color=colors[i],
                name='%.3f CI' % this_ci,
                legendgroup='%.3f CI' % this_ci
            ))

    # now draw the mean prediction and actuals
    if not interactive:
        plt.plot(
            y_hat.index,
            y_hat['predict_lead_%d_mean' % predict_ahead],
            color=colors[0],
            label='predicted')

        plt.plot(
            y_true.index,
            y_true,
            color='black',
            label='true')
    else:
        fig.add_trace(go.Scatter(
            x=y_hat.index,
            y=y_hat['predict_lead_%d_mean' % predict_ahead],
            line_color=colors[0],
            name='predicted'))

        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_true,
            line_color='black',
            name='true'))

    # now plot outliers
    if outlier_min_q is not None or outliers is not None:

        # if outliers is not passed, determine them through use of outlier_min_q
        if outlier_min_q is not None and outliers is None:
            valid_low = y_hat[these_cols[col_order[n_quants-1-(outlier_min_q-1)]]]
            valid_high = y_hat[these_cols[col_order[n_quants+(outlier_min_q-1)]]]
            outliers = (y_true > valid_high) | (y_true < valid_low)

        if not interactive:
            plt.plot(
                y_true[outliers].index,
                y_true[outliers],
                'o',
                ms=5,
                color='r',
                label='outlier')
        else:
            fig.add_trace(go.Scatter(
                x=y_true[outliers].index,
                y=y_true[outliers],
                mode='markers',
                marker={'color': 'red'},
                name='outlier'))

    # set some plot properties
    if data_prop is not None:
        data_range = y_true.max() - y_true.min()
        if y_range is None:
            ymax = y_true.max() + data_range * data_prop
            ymin = y_true.min() - data_range * data_prop
            y_range = [ymin, ymax]

    if not interactive:

        if title is not None:
            plt.title(title)
        plt.legend(loc='best')
        plt.ylabel(y_title)
        sns.despine(offset=10)
        if y_range is not None:
            plt.ylim(y_range)
        if date_range is not None:
            plt.xlim(date_range)

    else:
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
