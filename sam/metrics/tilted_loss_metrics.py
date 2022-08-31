from typing import Sequence

import numpy as np
import pandas as pd


def tilted_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5, axis: int = -1):
    r"""
    Calculate tilted (also known as pinball or quantile loss) with numpy. Given a quantile
    :math:`q`, and an error :math:`e=y_\text{true}-y_\text{pred}`, then tilted loss :math:`L_q`
    is defined as:

    .. math::
        L_q(e) = \sum_{i:e_i\leq0} (1 - q) |e_i| + \sum_{i:e_i > 0} q |e_i|\\
        \text{where } i \in \{0, \dots, n_\text{samples} - 1\} \text{ and } q \in [0, 1]

    This function is equivalent to the mean absolute error if :math:`q=0.5`, which approximates
    the median:

    .. math::
       \begin{eqnarray}
       L_\text{0.5} &= \sum_{i:e_i\leq0} 0.5 |e_i| + \sum_{i:e_i > 0} 0.5 |e_i| \\
                    &= 0.5 * \sum_i |e_i|\\
       \text{MAE} &= \frac{1}{n_\text{samples}} \sum_i |e_i|
       \text{MAE} &= 2 L_\text{0.5}
       \end{eqnarray}

    For a given value of :math:`q`, the function that minimizes the tilted loss will be the
    :math:`q`'th quantile function.

    Parameters
    ----------
    y_true: array-like of shape = (n_samples, n_cols)
        True labels.
    y_pred: array-like of shape = (n_samples, n_cols)
        Predictions. Must be same shape as `y_true`
    quantile: float, optional (default=0.5)
        The quantile to use when computing tilted loss.
        Must be between 0 and 1 for tilted loss to be positive.
    axis: int, optional (default=-1)
        Over which axis to take the mean. By default the last axis (-1) is taken. For a
        (n_samples, n_cols) target array set axis=0 to retrieve the tilted loss of each col.

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
    0.15000000000000002
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    e = y_true - y_pred

    return np.mean(np.maximum(quantile * e, (quantile - 1) * e), axis=axis)


def joint_mae_tilted_loss(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    quantiles: Sequence[float] = None,
    n_targets: int = 1,
):
    """
    Joint average and quantile regression loss function using mae.
    Sum of mean absolute error and multiple tilted loss functions
    Custom loss function, inspired by https://github.com/fmpr/DeepJMQR

    This calculates loss for multiple quantiles, and multiple targets.
    The total loss is the sum of the mae of all targets, and the tilted
    loss of all quantiles.

    ``y_true`` is expected to be a dataframe with shape ``(n_rows, n_targets)``
    ``y_pred`` is expected to be a dataframe with shape ``(n_rows, n_targets * (n_quantiles + 1))``
    For example, if there are 2 outputs and 2 quantiles, the order of y_pred
    should be: `[output_1_quantile_1, output_2_quantile_1,
    output_1_quantile_2, output_2_quantile_2, output_1_mean, output_2_mean]`

    Parameters
    ----------
    y_true: dataframe
        True values
    y_pred: dataframe
        Predicted values
    quantiles: sequence of floats (default=None)
        Quantiles to predict. Values between 0 and 1. By default, no quantile loss is used.
    n_targets: integer, optional (default=1)
        The number of distinct outputs to predict

    Returns
    -------
    float:
        The joint mae tilted loss

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sam.metrics import joint_mae_tilted_loss
    >>> y_true = pd.DataFrame(np.array([4, 5, 6]), columns=["output_1"])
    >>> y_pred = pd.DataFrame(
    ...     np.array([[1, 2, 3], [7, 8, 9], [3.9, 5.1, 5.9]]),
    ...     columns=["output_1_quantile_1", "output_1_quantile_2", "output_1_mean"],
    ... )
    >>> joint_mae_tilted_loss(y_true, y_pred, quantiles=[0.1, 0.9], n_targets=1)
    3.44
    """
    if quantiles is None:
        quantiles = []

    average_prediction = y_pred.iloc[:, -1 * n_targets :].values
    # The last node will be fit with 2x the 0.5 quantile tilted loss (same as MAE)
    loss = 2 * np.sum(tilted_loss(y_true, average_prediction, quantile=0.5, axis=0))
    # For each quantile fit one node with corresponding tilted loss
    for i, q in enumerate(quantiles):
        # Select the i-th node
        q_pred = y_pred.iloc[:, n_targets * i : n_targets * (i + 1)].values
        # add tilted loss to total loss
        loss += np.sum(tilted_loss(y_true, q_pred, quantile=q, axis=0))
    return loss


def joint_mse_tilted_loss(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    quantiles: Sequence[float] = None,
    n_targets: int = 1,
):
    """
    Joint average and quantile regression loss function using mse.
    Sum of mean absolute error and multiple tilted loss functions
    Custom loss function, inspired by https://github.com/fmpr/DeepJMQR

    This calculates loss for multiple quantiles, and multiple targets.
    The total loss is the sum of the mse of all targets, and the tilted
    loss of all quantiles.

    ``y_true`` is expected to be a dataframe with shape ``(n_rows, n_targets)``
    ``y_pred`` is expected to be a dataframe with shape ``(n_rows, n_targets * (n_quantiles + 1))``
    For example, if there are 2 outputs and 2 quantiles, the order of columns of y_pred
    should be: `[output_1_quantile_1, output_2_quantile_1,
    output_1_quantile_2, output_2_quantile_2, output_1_mean, output_2_mean]`

    Parameters
    ----------
    y_true: dataframe
        True values
    y_pred: dataframe
        Predicted values
    quantiles: sequence of floats (default=None)
        Quantiles to predict. Values between 0 and 1. By default, no quantile loss is used.
    n_targets: integer, optional (default=1)
        The number of distinct outputs to predict

    Returns
    -------
    float:
        The joint mse tilted loss

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sam.metrics import joint_mae_tilted_loss
    >>> y_true = pd.DataFrame(np.array([4, 5, 6]), columns=["output_1"])
    >>> y_pred = pd.DataFrame(
    ...     np.array([[1, 2, 3], [7, 8, 9], [3.9, 5.1, 5.9]]),
    ...     columns=["output_1_quantile_1", "output_1_quantile_2", "output_1_mean"],
    ... )
    >>> joint_mse_tilted_loss(y_true, y_pred, quantiles=[0.1, 0.9], n_targets=1)
    7.410000000000002
    """
    if quantiles is None:
        quantiles = []

    average_prediction = y_pred.iloc[:, -1 * n_targets :].values
    # The last node will be fit with regular mean squared error
    loss = np.sum(np.mean((y_true - average_prediction) ** 2, axis=0))
    # For each quantile fit one node with corresponding tilted loss
    for i, q in enumerate(quantiles):
        # Select the i-th node
        q_pred = y_pred.iloc[:, n_targets * i : n_targets * (i + 1)].values
        # add tilted loss to total loss
        loss += np.sum(tilted_loss(y_true, q_pred, quantile=q, axis=0))
    return loss
