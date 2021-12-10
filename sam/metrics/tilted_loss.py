import numpy as np


def tilted_loss(y_true: np.array, y_pred: np.array, quantile: float = 0.5):
    """
    Calculate tilted, or quantile loss with numpy. Given a quantile q, and an error e,
    then tilted loss is defined as `(1-q) * |e|` if `e < 0`, and `q * |e|` if `e > 0`.

    This function is the same as the mean absolute error if q=0.5, which approximates the median.
    For a given value of q, the function that minimizes the tilted loss will be the q'th quantile
    function.

    Parameters
    ----------
    y_true: array-like of shape = (n_samples)
        True labels.
    y_pred: array-like of shape = (n_samples)
        Predictions. Must be same shape as `y_true`
    quantile: float, optional (default=0.5)
        The quantile to use when computing tilted loss.
        Must be between 0 and 1 for tilted loss to be positive.

    Returns
    -------
    float:
        The quantile loss

    Examples
    --------
    >>> import numpy as np
    >>> from sam.metrics import tilted_loss
    >>> actual = np.array([1, 2, 3, 4])
    >>> pred = np.array([0.9, 2.1, 2.9, 3.1])
    >>> tilted_loss(actual, pred, quantile=0.5)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    e = y_true - y_pred
    return np.mean(np.maximum(quantile * e, (quantile - 1) * e), axis=-1)
