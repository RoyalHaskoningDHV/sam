import numpy as np
from sklearn.metrics import mean_absolute_error


def mean_absolute_scaled_error(
    y_true: np.array,
    y_pred: np.array,
    shift: int,
    sample_weight: np.array = None,
    multioutput: str = "uniform_average",
):
    """
    Given true value and predicted value, calculates MASE. Lower is better.
    MASE is the mean absolute error, divided by the MAE of a naive benchmark. The naive benchmark
    is the 'persistence benchmark': predicting the same value shift points ahead. For example,
    when `shift=1`, the naive benchmark is the MAE when for timestep `n`, we predict the value at
    timestep `n-1`. As shift increases, the naive benchmark will get worse, and the MASE will get
    better (lower).

    Even though this function does not require timestamps as input, it requires that
    the inputs are sorted by TIME (ascending), and that the timestamps are uniform. Otherwise, the
    output of this function will have no meaning.

    The shift points at the beginning of the array cannot be used for the persistence benchmark,
    but they are used for calculating the MAE of the prediction.
    If the truth vector is constant, the naive benchmark will be 0. In this case, the function
    will throw an error since MASE has no meaning.

    Parameters
    ----------
    y_true: array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values, sorted by time (ascending), with uniform timestamps
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values, sorted by time (ascending), with uniform timestamps
    shift: int
        The shift used when calculating the naive benchmark. For example, when the timestamps have
        a frequency of 15 minutes, and your model is predicting 1 hour ahead, then you should set
        shift to 4 to have an honest benchmark.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.

    multioutput : string in [‘raw_values’, ‘uniform_average’]
        or array-like of shape `(n_outputs)` Defines aggregating of multiple output values.

        'raw_values':
            Returns a full set of errors in case of multioutput input.

        'uniform_average':
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss: float or ndarray of floats
        MASE output is non-negative floating point. The best value is 0.0.

    Examples
    --------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [0.5, 1.5, 2.5]
    >>> # persistence benchmark would have a loss of 1. Our prediction has a loss of 0.5
    >>> mean_absolute_scaled_error(y_true, y_pred, shift=1)
    0.5
    """
    if shift != int(shift) or shift <= 0:
        raise ValueError("Shift must be a positive integer")
    # So we can accept pandas/python lists as well
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # If sample_weight is not None, we have to subscript it before passing to MAE
    naive_sampleweight = sample_weight[shift:] if sample_weight is not None else None
    naive_mae = mean_absolute_error(
        y_true[shift:],
        y_true[:-shift],
        sample_weight=naive_sampleweight,
        multioutput=multioutput,
    )
    actual_mae = mean_absolute_error(
        y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
    )

    if np.isscalar(naive_mae) and naive_mae == 0:
        raise ValueError("Target vector is constant. MASE has no meaning in this circumstance.")
    elif (naive_mae == 0).any():
        raise ValueError(
            "One of the target vectors is constant. MASE has no meaning in this circumstance."
        )
    return actual_mae / naive_mae
