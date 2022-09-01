import warnings
from typing import List

try:
    # Tensorflow often raises warnings when importing.
    # When importing sam, it is not necessary to show these warnings since they
    # are not relevant to sam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        import tensorflow as tf
        import tensorflow.keras.backend as K
except ImportError:
    # These are optional dependencies so it's not necessary to crash if they aren't found.
    # However, this will crash once one of the functions below runs.
    pass


def keras_tilted_loss(y_true: tf.Tensor, y_pred: tf.Tensor, quantile: float = 0.5):
    """
    Calculate tilted, or quantile loss in Keras. Given a quantile q, and an error e,
    tilted loss is defined as `(1-q) * |e|` if `e < 0`, and `q * |e|` if `e > 0`.

    This function is the same as the mean absolute error if q=0.5, which approximates the median.
    For a given value of q, the function that minimizes the tilted loss will be the q'th quantile
    function.

    Parameters
    ----------
    y_true: Theano/TensorFlow tensor.
        True labels.
    y_pred: Theano/TensorFlow tensor.
        Predictions. Must be same shape as `y_true`
    quantile: float, optional (default=0.5)
        The quantile to use when computing tilted loss.
        Must be between 0 and 1 for tilted loss to be positive.

    Examples
    --------
    >>> from sam.metrics import keras_tilted_loss
    >>> from sam.models import create_keras_quantile_mlp
    >>> n_input = 1000
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> quantile = 0.5  # Quantile, in this case the median
    >>> model.compile(loss=lambda y,f: keras_tilted_loss(y, f, quantile))
    """
    e = y_true - y_pred
    return K.mean(K.maximum(quantile * e, (quantile - 1) * e), axis=-1)


def keras_joint_mse_tilted_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    quantiles: List[float] = None,
    n_targets: int = 1,
):
    """
    Joint mean and quantile regression loss function using mse.
    Sum of mean squared error and multiple tilted loss functions
    Custom loss function, inspired by https://github.com/fmpr/DeepJMQR
    Only compatible with tensorflow backend.

    This calculates loss for multiple quantiles, and multiple targets.
    The total loss is the sum of the mse of all targets, and the tilted
    loss of all quantiles.

    `y_true` is expected to be a tensor with shape `(None, n_targets)`
    `y_pred` is expected to be a tensor with shape `(None, n_targets * (n_quantiles + 1))`
    For example, if there are 2 outputs and 2 quantiles, the order of y_pred
    should be: `[output_1_quantile_1, output_2_quantile_1,
    output_1_quantile_2, output_2_quantile_2, output_1_mean, output_2_mean]`

    Parameters
    ----------
    y_true: tensorflow tensor
        True values
    y_pred: tensorflow tensor
        Predicted values
    quantiles: list of floats (default=None)
        Quantiles to predict. Values between 0 and 1. By default, no quantile loss is used.
    n_targets: integer, optional (default=1)
        The number of distinct outputs to predict

    Examples
    --------
    >>> from sam.metrics import keras_joint_mse_tilted_loss as mse_tilted
    >>> from sam.models import create_keras_quantile_mlp
    >>> n_input = 1000
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> qs = [0.1, 0.9]
    >>> model.compile(loss=lambda y,f: mse_tilted(y, f, qs))
    """
    if quantiles is None:
        quantiles = []
    # select the last column (nodes) of the output
    k = len(quantiles)
    mean_pred = tf.slice(y_pred, [0, k * n_targets], [-1, n_targets])
    # The last node will be fit with regular mean squared error
    loss = K.sum(K.mean(K.square(y_true - mean_pred), axis=0), axis=-1)
    # For each quantile fit one node with corresponding tilted loss
    for k in range(len(quantiles)):
        q = quantiles[k]
        # Select the kth node
        q_pred = tf.slice(y_pred, [0, k * n_targets], [-1, n_targets])
        e = y_true - q_pred
        # add tilted loss to total loss
        loss += K.sum(K.mean(K.maximum(q * e, (q - 1) * e), axis=0), axis=-1)
    return loss


def keras_joint_mae_tilted_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    quantiles: List[float] = None,
    n_targets: int = 1,
):
    """Joint mean and quantile regression loss function using mae.
    Sum of mean absolute error and multiple tilted loss functions
    Custom loss function, inspired by https://github.com/fmpr/DeepJMQR
    Only compatible with tensorflow backend.

    This calculates loss for multiple quantiles, and multiple targets.
    The total loss is the sum of the mae of all targets, and the tilted
    loss of all quantiles.

    ``y_true`` is expected to be a tensor with shape ``(None, n_targets)``
    ``y_pred`` is expected to be a tensor with shape ``(None, n_targets * (n_quantiles + 1))``
    For example, if there are 2 outputs and 2 quantiles, the order of y_pred
    should be: `[output_1_quantile_1, output_2_quantile_1,
    output_1_quantile_2, output_2_quantile_2, output_1_mean, output_2_mean]`

    Parameters
    ----------
    y_true: tensorflow tensor
        True values
    y_pred: tensorflow tensor
        Predicted values
    quantiles: list of floats (default=None)
        Quantiles to predict. Values between 0 and 1. By default, no quantile loss is used.
    n_targets: integer, optional (default=1)
        The number of distinct outputs to predict

    Examples
    --------
    >>> from sam.metrics import keras_joint_mae_tilted_loss as mae_tilted
    >>> from sam.models import create_keras_quantile_mlp
    >>> n_input = 1000
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> qs = [0.1, 0.9]
    >>> model.compile(loss=lambda y,f: mae_tilted(y, f, qs))
    """
    if quantiles is None:
        quantiles = []
    # select the last column (nodes) of the output
    k = len(quantiles)
    mean_pred = tf.slice(y_pred, [0, k * n_targets], [-1, n_targets])
    # The last node will be fit with 0.5 quantile
    loss = K.sum(K.mean(K.abs(y_true - mean_pred), axis=0), axis=-1)
    # For each quantile fit one node with corresponding tilted loss
    for k in range(len(quantiles)):
        q = quantiles[k]
        # Select the kth node
        q_pred = tf.slice(y_pred, [0, k * n_targets], [-1, n_targets])
        e = y_true - q_pred
        # add tilted loss to total loss
        loss += K.sum(K.mean(K.maximum(q * e, (q - 1) * e), axis=0), axis=-1)
    return loss


def keras_rmse(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Calculate root mean squared error in Keras.

    Parameters
    ----------
    y_true: Theano/TensorFlow tensor.
        True labels.
    y_pred: Theano/TensorFlow tensor.
        Predictions. Must be same shape as `y_true`

    Examples
    --------
    >>> from sam.metrics import keras_rmse
    >>> from sam.models import create_keras_quantile_mlp
    >>> n_input = 1000
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> model.compile(loss=keras_rmse)
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_keras_forecasting_metrics(quantiles: List[float] = None):
    """
    Get list of standard forecasting metrics.
    These metrics could be used specifically for regression problems, such as forecasting.
    The default list is mse, mae, rmse. Quantile loss can also be added, by giving
    a list of quantiles.

    Parameters
    ----------
    quantiles: list of floats (default=None)
        List of quantiles to make losses of. Values between 0 and 1.
        By default, no quantile loss is used. Note that mae is already included and is
        the same as quantile loss with quantile 0.5.

    Examples
    --------
    >>> from sam.metrics import keras_rmse, get_keras_forecasting_metrics
    >>> from sam.models import create_keras_quantile_mlp
    >>> n_input = 1000
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> model.compile(loss=keras_rmse, metrics=get_keras_forecasting_metrics())
    """
    if quantiles is None:
        quantiles = []
    quantile_metrics = [lambda y, m: keras_tilted_loss(y, m, q) for q in quantiles]
    main_metrics = ["mse", "mae", keras_rmse]
    return main_metrics + quantile_metrics
