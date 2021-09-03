import sys
import warnings

import numpy as np
import pandas as pd


def train_r2(true, predicted, benchmark):
    """
    Calculation of r2 using a benchmark (often mean/median/mode of the train set). The idea is
    that calculating r2 over a an average of the test set is 'leaking' information. This is
    especially problematic if the test set is small compared to the train set. It is therefore
    more suitable to compare the model predictions to the average estimated in the train set.
    You can also define an array benchmark, for example a prediction using your model with as
    input a certain feature nulled (such as rainfall prediction). Then, by comparing this
    benchmark with your model prediction with a non-null rainfall prediction you can determine
    that that certain (rain) feature is important in your model and thus indeed improves
    predictions.

    Parameters
    ----------
    true: numpy array or pd.Series
        Actual timeseries
    predicted: numpy array or pd.Series
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
        assert true_ravel.size == true.shape[0], 'true argument must be 1 dimensional'
        true = true_ravel
    if isinstance(benchmark, (pd.Series, np.ndarray)) and len(benchmark.shape) > 1:
        benchmark_ravel = np.ravel(benchmark)
        assert benchmark_ravel.size == benchmark.shape[0], ('benchmark argument must be '
                                                            '1 dimensional')
        assert benchmark_ravel.size == true.size, 'benchmark array must be same size as true array'
        benchmark = benchmark_ravel
    if len(predicted.shape) > 1:
        predicted_ravel = np.ravel(predicted)
        assert predicted_ravel.size == predicted.shape[0], ('predicted argument must be '
                                                            '1 dimensional')
        predicted = predicted_ravel

    num = np.nansum((true - predicted)**2)
    denom = np.nansum((true - benchmark)**2)

    r2 = 1 - (num / (denom+sys.float_info.epsilon))

    return r2


def train_mean_r2(true, predicted, train_mean):
    """
    Calculation of r2 with custom mean, so you can pass the mean from the train set.
    The idea is that calculating r2 over a test set is 'leaking' information to the mean benchmark.
    This is especially problematic if the test set is small compared to the train set.
    It is therefore more suitable to compare the model predictions to the mean estimated in the
    train set.

    Parameters
    ----------
    true: numpy array or pd.Series
        Actual timeseries
    predicted: numpy array or pd.Series
        Predicted timeseries
    train_mean: float
        Mean estimated from the train set

    Returns:
    -------
    r2: float
        Custom R2 with training mean
    """
    warnings.warn("DEPRECATED: USE train_r2(true, predicted, benchmark)"
                  " WITH SAME PARAMETERS", DeprecationWarning)

    return train_r2(true, predicted, train_mean)
