from typing import Callable, Optional

from keras import Optimizer

from sam.metrics import keras_joint_mae_tilted_loss, keras_joint_mse_tilted_loss


def create_keras_quantile_mlp(
    n_input: int,
    n_neurons: int,
    n_layers: int,
    n_target: int = 1,
    quantiles: list = None,
    dropout: float = 0.0,
    momentum: float = 1.0,
    hidden_activation: str = "relu",
    output_activation: str = "linear",
    lr: float = 0.001,
    average_type: str = "mean",
    optimizer: Optional[Optimizer] = None,
) -> Callable:
    """
    Creates a multilayer perceptron in keras.
    Optimizes the keras_joint_mse_tilted_loss to do multiple quantile and
    mean/median regression with a single model.

    Parameters
    ----------
    n_input: int
        Number of input nodes
    n_neurons: int
        Number of neurons hidden layer
    n_layers: int
        Number of hidden layers. 0 implies that the output is no additional layer
        between input and output.
    n_target: int, optional (default=1)
        Number of distinct outputs. Each will have their own mean and quantiles
        When fitting the model, this should be equal to the number of columns in y_train
    quantiles: list of floats (default=None)
        Quantiles to predict, values between 0 and 1,
        default is None, which returns a regular mlp (single output)
        for mean squared error regression
    dropout: float, optional (default=0.0)
        Rate parameter for dropout, value in (0,1).
        Default is 0.0, which means that no batch dropout is applied
    momentum: float, optional (default=1.0)
        Parameter for batch normalization, value in (0,1)
        default is 1.0, which means that no batch normalization is applied
        Smaller values means stronger batch normalization, see keras documentation.
        https://keras.io/layers/normalization/
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layer, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate
    average_type: str (default='mean')
        Determines what to fit as the average: 'mean', or 'median'. The average is the last
        node in the output layer and does not reflect a quantile, but rather estimates the central
        tendency of the data. Setting to 'mean' results in fitting that node with MSE, and
        setting this to 'median' results in fitting that node with MAE (equal to 0.5 quantile).
    optimizer: Optimizer (default=None)
        Forcefully overwrites the default Adam optimizer object.

    Returns
    --------
        keras model

    Examples
    --------
    >>> from sam.models import create_keras_quantile_mlp
    >>> from sam.datasets import load_rainbow_beach
    >>> data = load_rainbow_beach()
    >>> X, y = data, data["water_temperature"]
    >>> n_input = X.shape[1]
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles=quantiles)
    >>> model.fit(X, y, batch_size=16, epochs=20, verbose=0)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    """
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Dense,
        Dropout,
        Input,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if quantiles is None:
        quantiles = []
    if dropout is None:
        dropout = 0.0
    if momentum is None:
        momentum = 1.0
    if len(quantiles) == 0:
        mse_tilted = "mse"
    else:

        def mse_tilted(y, f):
            if average_type == "mean":
                loss = keras_joint_mse_tilted_loss(y, f, quantiles, n_target)
            elif average_type == "median":
                loss = keras_joint_mae_tilted_loss(y, f, quantiles, n_target)
            return loss

    n_out = n_target * (len(quantiles) + 1)  # one extra for mean regression
    input_layer = Input((n_input,))
    h = input_layer

    # repeat adding the same type of layer
    for _ in range(n_layers):
        h = Dense(n_neurons)(h)
        if momentum < 1:
            h = BatchNormalization(momentum=momentum)(h)
        h = Activation(hidden_activation)(h)
        h = Dropout(rate=dropout)(h)
    out = Dense(n_out, activation=output_activation)(h)

    model = Model(inputs=input_layer, outputs=out)
    optimizer = Adam(learning_rate=lr) if optimizer is None else optimizer
    model.compile(loss=mse_tilted, optimizer=optimizer)
    return model


def create_keras_quantile_rnn(
    input_shape: tuple,
    n_neurons: int = 64,
    n_layers: int = 2,
    quantiles: list = None,
    n_target: int = 1,
    layer_type: str = "GRU",
    dropout: float = 0.0,
    recurrent_dropout: str = "dropout",
    hidden_activation: str = "relu",
    output_activation: str = "linear",
    lr: float = 0.001,
    optimizer: Optional[Optimizer] = None,
) -> Callable:
    """
    Creates a simple RNN (LSTM or GRU) with keras.
    Optimizes the keras_joint_mse_tilted_loss to do multiple quantile and
    mean regression with a single model.

    Parameters
    ----------
    input_shape: tuple,
        A shape tuple (integers) of single input sample, (window, number of features)
        where window is the parameter used in the preprocessing.RecurrentReshaper class
    n_neurons: int (default=64)
        Number of neurons hidden layer
    n_layers: int (default=2)
        Number of hidden layers. 0 implies that the output is no additional layer
        between input and output.
    quantiles: list of floats (default=None)
        Quantiles to predict, values between 0 and 1,
        default is None, which returns a regular rnn (single output)
        for mean squared error regression
    n_target: int, optional (default=1)
        Number of distinct outputs. Each will have their own mean and quantiles
        When fitting the model, this should be equal to the number of columns in y_train
    layer_type: str (default='GRU')
        Type of recurrent layer
        Options: 'LSTM' (long short-term memory) or 'GRU' (gated recurrent unit)
    dropout: float, optional (default=0.0)
        Rate parameter for dropout, value in (0,1)
        default is 0.0, which means that no batch dropout is applied
    recurrent_dropout: float or str, optional (default='dropout')
        Rate parameter for dropout, value in (0,1)
        default is 'dropout', which means that recurrent dropout is equal to
        dropout parameter (dropout between layers)
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layer, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate
    optimizer: Optimizer (default=None)
        Forcefully overwrites the default Adam optimizer object.

    Returns
        keras model

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sam.data_sources import synthetic_date_range, synthetic_timeseries
    >>> from sam.preprocessing import RecurrentReshaper
    >>> from sam.models import create_keras_quantile_rnn
    >>> dates = pd.Series(synthetic_date_range().to_pydatetime())
    >>> y = synthetic_timeseries(dates, daily = 2, noise = {'normal': 0.25}, seed=2)
    >>> y = y[~np.isnan(y)]
    >>> X = pd.DataFrame(y)
    >>> X_3d = RecurrentReshaper(window=24, lookback = 1).fit_transform(X)
    >>> X_3d = X_3d[24:]
    >>> y = y[24:]
    >>> input_shape = X_3d.shape[1:]
    >>> model = create_keras_quantile_rnn(input_shape, quantiles=[0.01, 0.99])
    >>> model.fit(X_3d, y, batch_size=32, epochs=5, verbose=0)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    """
    from tensorflow.keras.layers import GRU, LSTM, Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if quantiles is None:
        quantiles = []
    if dropout is None:
        dropout = 0.0
    if len(quantiles) == 0:
        mse_tilted = "mse"
    else:

        def mse_tilted(y, f):
            loss = keras_joint_mse_tilted_loss(y, f, quantiles, n_target)
            return loss

    # Recurrent layer is either LSTM or GRU
    if layer_type.upper() == "LSTM":
        RNN = LSTM
    elif layer_type.upper() == "GRU":
        RNN = GRU
    else:
        raise ValueError('Invalid layer_type, choose "LSTM" or "GRU"')
    if recurrent_dropout == "dropout":
        recurrent_dropout = dropout
    n_out = n_target * (len(quantiles) + 1)  # one extra for mean regression

    input_layer = Input(input_shape)
    h = input_layer
    for i in range(n_layers):
        # last layer returns last value of sequence:
        return_sequences = i < n_layers - 1
        h = RNN(
            n_neurons,
            activation=hidden_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences,
        )(h)
    out = Dense(n_out, activation=output_activation)(h)
    model = Model(inputs=input_layer, outputs=out)
    optimizer = Adam(learning_rate=lr) if optimizer is None else optimizer
    model.compile(loss=mse_tilted, optimizer=optimizer)
    return model


def create_keras_autoencoder_mlp(
    n_input: int,
    encoder_neurons: list = [64, 16],
    dropout: float = 0.0,
    momentum: float = 1.0,
    hidden_activation: str = "relu",
    output_activation: str = "linear",
    lr: float = 0.001,
    optimizer: Optional[Optimizer] = None,
) -> Callable:
    """
    Function to create an MLP auto-encoder in keras
    Optimizes the mean squared error to reconstruct input,
    after passing input through bottleneck neural network.

    Parameters
    ----------
    n_input: int
        Number of input nodes
    encoder_neurons: list (default=[64, 16])
        List of integers, each representing the number of neurons per layer
        within the encoder. Decoder is reversed version of encoder.
        Last element is number of neurons in "representation" layer
        Example:
        If n_layers=[64, 12], number of features is 120,
        the number of neurons per layer is [120, 64, 12, 64, 120].
    dropout: float, optional (default=0.0)
        Rate parameter for dropout, value in (0,1)
        default is 0.0, which means that no batch dropout is applied
    momentum: float, optional (default=1.0)
        Parameter for batch normalization, value in (0,1)
        default is 1.0, which means that no batch normalization is applied
        Smaller values means stronger batch normalization, see keras documentation.
        https://keras.io/layers/normalization/
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layers, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate
    optimizer: Optimizer (default=None)
        Forcefully overwrites the default Adam optimizer object.

    Returns
        keras model

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.data_sources import synthetic_date_range, synthetic_timeseries
    >>> from sam.preprocessing import RecurrentReshaper
    >>> from sam.models import create_keras_autoencoder_mlp
    >>> dates = pd.Series(synthetic_date_range().to_pydatetime())
    >>> X = [synthetic_timeseries(dates, daily=2, noise={'normal': 0.25}, seed=i)
    ...      for i in range(100)]
    >>> X = pd.DataFrame(X)
    >>> model = create_keras_autoencoder_mlp(n_input=100)
    >>> model.fit(X.T, X.T, batch_size=32, epochs=5, verbose=0)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    """
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Dense,
        Dropout,
        Input,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if dropout is None:
        dropout = 0.0
    if momentum is None:
        momentum = 1.0

    input_layer = Input((n_input,))
    # encoder
    h = input_layer
    # For each n_neuron value, add a dense layer to the model
    n_neurons = encoder_neurons + encoder_neurons[:-1][::-1]
    for n in n_neurons:
        h = Dense(n)(h)
        if momentum < 1:
            h = BatchNormalization(momentum=momentum)(h)
        h = Activation(hidden_activation)(h)
        h = Dropout(rate=dropout)(h)
    # output layer
    out = Dense(n_input, activation=output_activation)(h)
    # compile
    model = Model(inputs=input_layer, outputs=out)
    optimizer = Adam(learning_rate=lr) if optimizer is None else optimizer
    model.compile(loss="mse", optimizer=optimizer)
    return model


def create_keras_autoencoder_rnn(
    input_shape: tuple,
    encoder_neurons: list = [64, 16],
    layer_type: str = "GRU",
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    hidden_activation: str = "relu",
    output_activation: str = "linear",
    lr: float = 0.001,
    optimizer: Optional[Optimizer] = None,
) -> Callable:
    """
    Function to create a recurrent auto-encoder in keras
    Optimizes the mean squared error to reconstruct input,
    after passing input through bottleneck neural network.
    Reference:
    https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
    https://blog.keras.io/building-autoencoders-in-keras.html

    Parameters
    ----------
    input_shape: tuple,
        shape of single input sample, (recurrent steps, number of features)
    encoder_neurons: list (default=[64, 16])
        List of integers, each representing the number of neurons per layer
        within the encoder. Decoder is reversed version of encoder.
        Last element is number of neurons in "representation" layer
        Example:
        If n_layers=[64, 12], number of features is 120,
        the number of neurons per layer is [120, 64, 12, 64, 120].
    layer_type: str (default='GRU')
        Type of recurrent layer
        Options: 'LSTM' (long short-term memory) or 'GRU' (gated recurrent unit)
    dropout: float, optional (default=0.0)
        Rate parameter for dropout, value in (0,1)
        default is 0.0, which means that no batch dropout is applied
    recurrent_dropout: float, optional (default=0.0)
        Rate parameter for dropout, value in (0,1)
        default is 0.0, which means that recurrent dropout is equal to
        dropout parameter (dropout between layers)
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layers, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate
    optimizer: Optimizer (default=None)
        Forcefully overwrites the default Adam optimizer object.

    Returns
        keras model

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.data_sources import synthetic_date_range, synthetic_timeseries
    >>> from sam.preprocessing import RecurrentReshaper
    >>> from sam.models import create_keras_autoencoder_rnn
    >>> dates = pd.Series(synthetic_date_range().to_pydatetime())
    >>> y = synthetic_timeseries(dates, daily=2, noise={'normal': 0.25}, seed=2)
    >>> X = pd.DataFrame(y)
    >>> X_3d = RecurrentReshaper(window=24, lookback=1).fit_transform(X)
    >>> X_3d = X_3d[24:]
    >>> input_shape = X_3d.shape[1:]
    >>> model = create_keras_autoencoder_rnn(input_shape)
    >>> model.fit(X_3d, X_3d, batch_size=32, epochs=5, verbose=0)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    """
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        Dense,
        Input,
        RepeatVector,
        TimeDistributed,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if dropout is None:
        dropout = 0.0
    lookback = input_shape[0]
    n_features = input_shape[1]
    if layer_type.upper() == "LSTM":
        RNN = LSTM
    elif layer_type.upper() == "GRU":
        RNN = GRU
    else:
        raise ValueError('layer_type should be "LSTM" or "GRU"')

    input_layer = Input(input_shape)
    # encoder
    h = input_layer
    # For each n_neuron value, add an rnn layer to the model
    n_layers = len(encoder_neurons)
    for i in range(n_layers - 1):
        h = RNN(
            encoder_neurons[i],
            activation=hidden_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
        )(h)
    # Last layer of encoder: only return last value of the LSTM output sequence
    h = RNN(
        encoder_neurons[-1],
        activation=hidden_activation,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )(h)
    # decoder
    # repeat 1d representation to feed to an rnn layer
    h = RepeatVector(lookback)(h)
    # Reverse layers to create symmetric model
    for i in range(n_layers - 1, 0, -1):
        h = RNN(
            encoder_neurons[i],
            activation=hidden_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
        )(h)
    out = TimeDistributed(Dense(n_features, activation=output_activation))(h)
    # compile
    model = Model(inputs=input_layer, outputs=out)
    optimizer = Adam(learning_rate=lr) if optimizer is None else optimizer
    model.compile(loss="mse", optimizer=optimizer)
    return model
