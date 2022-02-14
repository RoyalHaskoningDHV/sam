import sys
import warnings
from typing import Union

import numpy as np
import pandas as pd


def train_r2(
    true: Union[pd.Series, np.array],
    predicted: Union[pd.Series, np.array],
    benchmark: Union[float, np.array],
):
    """
    Calculation of r2 using a benchmark (often mean/median/mode of the train set). The idea is
    that calculating r2 over a an average of the test set is 'leaking' information. This is
    especially problematic if the test set is small compared to the train set. It is therefore
    more suitable to compare the model predictions to the average estimated in the train set.
    It is also possible to define an array benchmark, for example a prediction using your model
    with input a certain feature nulled (such as rainfall prediction). Then, by comparing this
    benchmark with the model prediction with a non-null rainfall prediction it's possible to
    determine that certain (for example rain) feature is important in the model and thus
    indeed improves predictions.

    Parameters
    ----------
    true: np.array or pd.Series
        Actual timeseries
    predicted: np.array or pd.Series
        Predicted timeseries
    benchmark: float or pd.Series
        Average (mean/median/mode) estimated from the train set or a timeseries to compare to

    Returns:
    -------
    r2: float
        Custom R2 with training mean/median
    """

    if len(true.shape) > 1:
        true_ravel = np.ravel(true)
        if true_ravel.size > true.shape[0]:
            raise ValueError("true argument must be 1 dimensional")

        true = true_ravel
    if isinstance(benchmark, (pd.Series, np.ndarray)) and len(benchmark.shape) > 1:
        benchmark_ravel = np.ravel(benchmark)
        if benchmark_ravel.size > benchmark.shape[0]:
            raise ValueError("benchmark argument must be 1 dimensional")
        if benchmark_ravel != true.size:
            raise ValueError("benchmark array must be same size as true array")
        benchmark = benchmark_ravel
    if len(predicted.shape) > 1:
        predicted_ravel = np.ravel(predicted)
        if predicted_ravel.size > predicted.shape[0]:
            raise ValueError("predicted argument must be 1 dimensional")
        predicted = predicted_ravel

    num = np.nansum((true - predicted) ** 2)
    denom = np.nansum((true - benchmark) ** 2)

    r2 = 1 - (num / (denom + sys.float_info.epsilon))

    return r2


def train_mean_r2(
    true: Union[pd.Series, np.array],
    predicted: Union[pd.Series, np.array],
    train_mean: float,
):
    """
    Calculation of r2 with custom mean, so you can pass the mean from the train set.
    The idea is that calculating r2 over a test set is 'leaking' information to the mean benchmark.
    This is especially problematic if the test set is small compared to the train set.
    It is therefore more suitable to compare the model predictions to the mean estimated in the
    train set.

    Parameters
    ----------
    true: np.array or pd.Series
        Actual timeseries
    predicted: np.array or pd.Series
        Predicted timeseries
    train_mean: float
        Mean estimated from the train set

    Returns:
    -------
    r2: float
        Custom R2 with training mean
    """
    warnings.warn(
        "DEPRECATED: USE train_r2(true, predicted, benchmark)" " WITH SAME PARAMETERS",
        DeprecationWarning,
    )

    return train_r2(true, predicted, train_mean)
