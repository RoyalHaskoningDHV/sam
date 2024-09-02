import sys
from typing import Callable

import numpy as np
import pandas as pd
from sam.metrics import train_r2
from sklearn.metrics import mean_absolute_error


def performance_evaluation_fixed_predict_ahead(
    y_true_train: pd.Series,
    y_hat_train: pd.DataFrame,
    y_true_test: pd.Series,
    y_hat_test: pd.DataFrame,
    resolutions: list = [None],
    predict_ahead: int = 0,
    train_avg_func: Callable = np.nanmean,
    metric: str = "R2",
):
    """
    This function evaluates model performance over time for a single given predict ahead.
    It plots and returns r-squared, and creates a scatter plot of prediction vs true values.
    It does so for different temporal resolutions and for both test and train sets.
    The idea of evaluating performance for different time resolutions is that it gives
    insight into the resolution of the underlying patterns that the model is capturing
    (e.g. if the data has a very high time-resolution, the model might not capture every
    minute-to-minute change, but it could capture the hourly or daily patterns).
    Using this approach it is possible to find out what the best time-resolution is and use
    the result in, for instance, the sam.visualizations.quantile_plot.sam_quantile_plot.

    Parameters
    ----------
    y_true_train: pd.Series
        Series that contains the true train values.
    y_hat_train: pd.DataFrame
        DataFrame that contains the predicted train values (output of
        `MLPTimeseriesRegressor.predict` method)
    y_true_test: pd.Series
        Series that contains the true test values.
    y_hat_test: pd.DataFrame
        DataFrame that contains the predicted test values (output of
        `MLPTimeseriesRegressor.predict` method)
    resolutions: list (default=[None])
        List of strings (and/or None) that are interpretable by pandas resampler.
        If set to None, will return results for the native data resolution.
        Valid options are e.g.: [None], [None, '15min', '1h'], or ['1h', '1d']
    predict_ahead: int (default=0)
        Predict_ahead to display performance for
    train_avg_func: func - Callable (default=np.nanmean)
        Optional argument to pass function to calculate the train set average, by default the
        mean is used. This average is used for calculating the r2 metric.
    metric: str (default='R2')
        Optional argument to define the metric to evaluate the performance of the train and test
        set. Options:
        - 'R2': conventional r2, best used for regression
        - 'MAE': mean absolute error, best used for forecasting
        By default the 'R2' metric is used.

    Returns
    ------
    metric_df: pd.DataFrame
        Dataframe that contains the metric (r2 or mae) values per test/train set and resolution
        combination. Contains columns: ['R2' or 'MAE', 'dataset', 'resolution']
    bar_fig: matplotlib figure
        Figure object that displays the metrics for each resolution and data set.
    scatter_fig: matplotlib figure
        Figure object that displays predicted vs true data for the different resolutions.
    best_res: string
        The resolution with the maximum metric value in the train set.

    Example
    -------

    # assuming you have some y_true_train, y_true_test and predictions y_hat_train and y_hat_test:
    >>> from sam.datasets import load_rainbow_beach
    >>> from sam.feature_engineering import SimpleFeatureEngineer
    >>> from sam.models import MLPTimeseriesRegressor
    >>> from sam.visualization import performance_evaluation_fixed_predict_ahead
    ...
    >>> data = load_rainbow_beach()
    >>> X, y = data, data["water_temperature"]
    >>> test_size = int(X.shape[0] * 0.33)
    >>> train_size = X.shape[0] - test_size
    >>> X_train, y_train = X.iloc[:train_size, :], y[:train_size]
    >>> X_test, y_test = X.iloc[train_size:, :], y[train_size:]
    ...
    >>> simple_features = SimpleFeatureEngineer(
    ...     rolling_features=[
    ...         ("wave_height", "mean", 24),
    ...         ("wave_height", "mean", 12),
    ...     ],
    ...     time_features=[
    ...         ("hour_of_day", "cyclical"),
    ...     ],
    ...     keep_original=False,
    ... )
    ...
    >>> model = MLPTimeseriesRegressor(
    ...     predict_ahead=(0,1),
    ...     feature_engineer=simple_features,
    ...     verbose=0,
    ... )
    ...
    >>> model.fit(X_train, y_train)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    >>> y_hat_train = model.predict(X_train)
    >>> y_hat_test = model.predict(X_test)
    >>> r2_df, bar_fig, scatter_fig, best_res = performance_evaluation_fixed_predict_ahead(
    ...     y_train,
    ...     y_hat_train,
    ...     y_test,
    ...     y_hat_test,
    ...     resolutions=[None, '15min', '1h', '3h', '6h', '1h'])

    >>> # display the results
    >>> bar_fig.show()
    >>> scatter_fig.show()
    >>> print('best resolution found at %s'%best_res)  # doctest: +SKIP
    >>> r2_df.head()  # doctest: +SKIP
    >>> # print some results
    >>> best_res_r2 = r2_df.loc[(r2_df['dataset']=='train') &
    ...                         (r2_df['resolution'] == best_res), 'R2'].values[0]
    >>> native_r2 = r2_df.loc[(r2_df['dataset']=='train') &
    ...                       (r2_df['resolution'] == 'native'), 'R2'].values[0]
    >>> print('best resolution found at %s (%.3f vs %.3f native)'%(
    ...         best_res, best_res_r2, native_r2))  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # initialize scatter figure
    scatter_fig = plt.figure(figsize=(len(resolutions) * 3, 6))

    # select and shift the requested predict ahead
    y_hat_train = y_hat_train[f"predict_lead_{predict_ahead}_mean"].shift(predict_ahead)
    y_hat_test = y_hat_test[f"predict_lead_{predict_ahead}_mean"].shift(predict_ahead)

    # compute the metrics for the different temporal resolutions, for train and test data
    metric_list, dataset_list, resolution_list = [], [], []
    for ri, res in enumerate(resolutions):
        # resample data to desired resolution of requested
        if res is not None:
            res_label = res
            y_true_train_res = y_true_train.resample(res).mean()
            y_hat_train_res = y_hat_train.resample(res).mean()
            y_true_test_res = y_true_test.resample(res).mean()
            y_hat_test_res = y_hat_test.resample(res).mean()
        else:
            res_label = "native"
            y_true_train_res = y_true_train
            y_hat_train_res = y_hat_train
            y_true_test_res = y_true_test
            y_hat_test_res = y_hat_test

        # train set performance
        metric_list = _evaluate_performance(
            y_true_train_res,
            y_hat_train_res,
            y_true_train_res,
            train_avg_func,
            metric,
            metric_list,
        )

        # test set performance
        metric_list = _evaluate_performance(
            y_true_test_res,
            y_hat_test_res,
            y_true_train_res,
            train_avg_func,
            metric,
            metric_list,
        )

        # append results to lists
        dataset_list.append("train")
        resolution_list.append(res_label)
        dataset_list.append("test")
        resolution_list.append(res_label)

        # create scatter plot of train results:
        alpha = np.min([1000 / len(y_true_train_res), 1])
        plt.subplot(2, len(resolutions), ri + 1)
        ymin = np.min([y_true_train.min(), y_hat_train.min()])
        ymax = np.max([y_true_train.max(), y_hat_train.max()])
        plt.plot([ymin, ymax], [ymin, ymax], c="gray", ls="--")
        plt.plot(y_true_train_res.values, y_hat_train_res.values, "o", alpha=alpha)
        plt.title("train " + res_label)
        plt.xlim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if ri > 0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel("true")
            plt.ylabel("predicted")

        # create scatter plot of test results:
        plt.subplot(2, len(resolutions), ri + 1 + len(resolutions))
        plt.plot([ymin, ymax], [ymin, ymax], c="gray", ls="--")
        plt.plot(
            y_true_test_res.values,
            y_hat_test_res.values,
            "o",
            alpha=alpha,
            color="orange",
        )
        plt.title("test " + res_label)
        plt.xlim(ymin, ymax)
        plt.ylim(ymin, ymax)
        if ri > 0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel("true")
            plt.ylabel("predicted")

    # options for scatter plot
    sns.despine()
    plt.tight_layout()

    # create bar plot of different metrics
    metric_df = pd.DataFrame(
        {metric: metric_list, "dataset": dataset_list, "resolution": resolution_list}
    )
    bar_fig = plt.figure(figsize=(6, 4))
    plt.axhline(0, c="k")
    sns.barplot(data=metric_df, x="resolution", y=metric, hue="dataset")
    if metric == "R2":
        plt.ylabel("Variance Explained (%)")
        plt.ylim(0, 100)
    elif metric == "MAE":
        plt.ylabel("Mean Absolute Error (mm)")
    sns.despine()

    # calculate best resolution as the optimal resolution metric in the train set
    if metric == "R2":
        best_res = metric_df.iloc[metric_df.loc[metric_df["dataset"] == "train", metric].idxmax()][
            "resolution"
        ]
    elif metric == "MAE":
        best_res = metric_df.iloc[metric_df.loc[metric_df["dataset"] == "train", metric].idxmin()][
            "resolution"
        ]

    return metric_df, bar_fig, scatter_fig, best_res


def _evaluate_performance(
    y_true: pd.Series,
    y_hat: pd.Series,
    y_benchmark: pd.Series = None,
    avg_func: Callable = np.nanmean,
    metric: str = "R2",
    metric_list: list = None,
):
    """
    This function evaluates model performance using the specified metric.

    Parameters
    ----------
    y_true: pd.Series
        Series that contains the true values.
    y_hat: pd.Series
        Series that contains the predicted values.
    y_benchmark: pd.Series (default=None)
        Series that serves as a benchmark for r2 evaluation, required for calculating the r2 metric
    avg_func: func - Callable (default=np.nanmean)
        Optional argument to pass function to calculate the y_benchmark average, by default the
        mean is used. This average function is required for calculating the r2 metric.
    metric: str (default='R2')
        Optional argument to define the metric to evaluate the performance of the train and test
        set. Options:
        - 'R2': conventional r2, best used for regression
        - 'MAE': mean absolute error, best used for forecasting
        By default the 'R2' metric is used.
    metric_list: list (default=None)
        Optional argument to define a list that is appended with the performance results.

    Returns
    ------
    metric_list: list
        List with performance result
    """

    if metric_list is None:
        metric_list = []

    if metric == "R2":
        if y_benchmark is None:
            raise ValueError("y_benchmark needs to be supplied")
        else:
            # compute r2 with custom r2 function (in sam.metrics)
            try:
                benchmark = avg_func(y_benchmark)
            except Exception:
                e = sys.exc_info()[0]
                raise ValueError(f"Supplied avg_func resulted in error: {e}")

        r2 = train_r2(y_true, y_hat, benchmark)
        metric_list.append(r2 * 100)
    elif metric == "MAE":
        illegal_idx = y_true.isin([np.nan, -np.inf, np.inf]) | y_hat.isin(
            [np.nan, -np.inf, np.inf]
        )
        mae = mean_absolute_error(y_true[~illegal_idx], y_hat[~illegal_idx])
        metric_list.append(mae)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    return metric_list
