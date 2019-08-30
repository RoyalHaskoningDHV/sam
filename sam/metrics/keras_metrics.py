import tensorflow.keras.backend as K
import tensorflow as tf


def keras_tilted_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate tilted, or quantile loss in Keras. Given a quantile q, and an error e,
    then tilted loss is defined as (1-q) * |e| if e < 0, and q * |e| if e > 0.

    This function is the same as the mean absolute error if q=0.5, which approximates the median.
    For a given value of q, the function that minimizes the tilted loss will be the q'th quantile
    function.

    Parameters
    ----------
    y_true: Theano/TensorFlow tensor.
        True labels.
    y_pred: Theano/TensorFlow tensor.
        Predictions. Must be same shape as y_true
    quantile: float, optional (default=0.5)
        The quantile to use when computing tilted loss.
        Must be between 0 and 1 for tilted loss to be positive.

    Examples
    --------
    >>> from sam.metrics import keras_tilted_loss
    >>> model = Sequential(...)  # Any keras model
    >>> quantile = 0.5  # Quantile, in this case the median
    >>> model.compile(loss=lambda y,f: keras_tilted_loss(y, f, quantile))
    """
    e = y_true - y_pred
    return K.mean(K.maximum(quantile*e, (quantile-1)*e), axis=-1)


def keras_joint_mse_tilted_loss(y_true, y_pred, quantiles=[]):
    """ Joint mean and quantile regression loss function
    Sum of mean squared error and multiple tilted loss functions
    Custom loss function, inspired by https://github.com/fmpr/DeepJMQR
    Only compatible with tensorflow backend

    Parameters
    ----------
    y_true: tensorflow tensor
        True values
    y_pred: tensorflow tensor
        Predicted values
    quantiles: list of floats
        Quantiles to predict, values in (0,1)

    Examples
    --------
    >>> from sam.metrics import keras_joint_mse_tilted_loss as mse_tilted
    >>> model = Sequential(...)  # Any keras model
    >>> qs = [0.1, 0.9]
    >>> model.compile(loss=lambda y,f: mse_tilted(y, f, qs))
    """
    # select the last column (node) of the output
    k = len(quantiles)
    mean_pred = tf.slice(y_pred, [0, k], [-1, 1])
    # The last node will be fit with regular mean squared error
    loss = K.mean(K.square(y_true - mean_pred), axis=-1)
    # For each quantile fit one node with corresponding tilted loss
    for k in range(len(quantiles)):
        q = quantiles[k]
        # Select the kth node
        q_pred = tf.slice(y_pred, [0, k], [-1, 1])
        e = y_true - q_pred
        # add tilted loss to total loss
        loss += K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
    return(loss)


def keras_rmse(y_true, y_pred):
    """
    Calculate root mean squared error in Keras.

    Parameters
    ----------
    y_true: Theano/TensorFlow tensor.
        True labels.
    y_pred: Theano/TensorFlow tensor.
        Predictions. Must be same shape as y_true

    Examples
    --------
    >>> from sam.metrics import keras_rmse
    >>> model = Sequential(...)  # Any keras model
    >>> model.compile(loss=keras_rmse)
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_keras_forecasting_metrics(quantiles=[]):
    """
    List of standard forecasting metrics.
    These metrics could be used specifically for regression problems, such as forecasting.
    The default list is mse, mae, rmse. Quantile loss can also be added, by giving
    a list of quantiles.

    Parameters
    ----------
    quantiles: list
        list of quantiles to make losses of. By default, no quantile loss is used.
        Note that mae is already included and is the same as quantile loss with
        quantile 0.5.

    Examples
    --------
    >>> from sam.metrics import keras_rmse, keras_forecasting_metrics
    >>> model = Sequential(...)  # Any keras model
    >>> model.compile(loss=keras_rmse, metrics=keras_forecasting_metrics)
    """
    quantile_metrics = [lambda y, m: keras_tilted_loss(y, m, q) for q in quantiles]
    main_metrics = ['mse', 'mae', keras_rmse]
    return main_metrics + quantile_metrics
