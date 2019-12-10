import numpy as np
from sam.metrics import keras_joint_mse_tilted_loss


def create_keras_quantile_mlp(n_input,
                              n_neurons,
                              n_layers,
                              quantiles=[],
                              dropout=None,
                              momentum=None,
                              hidden_activation='relu',
                              output_activation='linear',
                              lr=0.001):
    """ Function to create a multilayer perceptron in keras
    Optimizes the keras_joint_mse_tilted_loss to do multiple quantile and
    mean regression with a single model.

    Parameters
    ----------
    n_input: int
        Number of input nodes
    n_neurons: int (default=None)
        Number of neurons hidden layer
    n_layers: int
        Number of hidden layers. 0 implies that the output is no additional layer
        between input and output.
    quantiles: list of floats (default=[])
        Quantiles to predict, values between 0 and 1,
        default is [], which returns a regular mlp (single output)
        for mean squared error regression
    dropout: float, optional (default=None)
        Rate parameter for dropout, value in (0,1)
        default is None, which means that no batch dropout is applied
    momentum: float, optional (default=None)
        Parameter for batch normalization, value in (0,1)
        default is None, which means that no batch normalization is applied
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layer, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate

    Returns
        keras model

    Examples
    --------
    >>> from sam.models import create_keras_quantile_mlp
    >>> from keras.datasets import boston_housing
    >>> (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    >>> n_input = x_train.shape[1]
    >>> n_neurons = 64
    >>> n_layers = 3
    >>> dropout = 0.2
    >>> quantiles = [0.1, 0.5, 0.9]
    >>> model = create_keras_quantile_mlp(n_input, n_neurons, n_layers, quantiles, dropout)
    >>> model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=20)
    """
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if len(quantiles) == 0:
        mse_tilted = 'mse'
    else:
        def mse_tilted(y, f):
            loss = keras_joint_mse_tilted_loss(y, f, quantiles)
            return(loss)

    n_out = len(quantiles) + 1  # one extra for mean regression
    input_layer = Input((n_input, ))
    h = input_layer

    # repeat adding the same type of layer
    for _ in range(n_layers):
        h = Dense(n_neurons)(h)
        if momentum is not None:
            h = BatchNormalization(momentum=momentum)(h)
        h = Activation(hidden_activation)(h)
        if dropout is not None:
            h = Dropout(rate=dropout)(h)
    out = Dense(n_out, activation=output_activation)(h)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(loss=mse_tilted,
                  optimizer=Adam(lr=lr))

    return(model)


def create_keras_quantile_rnn(input_shape,
                              n_neurons=64,
                              n_layers=2,
                              quantiles=[],
                              n_target=1,
                              layer_type='LSTM',
                              dropout=0.0,
                              recurrent_dropout='dropout',
                              hidden_activation='tanh',
                              output_activation='linear',
                              lr=0.001):
    """ Function to create a simple RNN (LSTM or GRU) with keras
    Optimizes the keras_joint_mse_tilted_loss to do multiple quantile and
    mean regression with a single model.

    Parameters
    ----------
    input_shape: tuple,
        shape of input X, (recurrent steps, number of features)
    n_neurons: int (default=None)
        Number of neurons hidden layer
    n_layers: int
        Number of hidden layers. 0 implies that the output is no additional layer
        between input and output.
    quantiles: list of floats (default=[])
        Quantiles to predict, values between 0 and 1,
        default is [], which returns a regular mlp (single output)
        for mean squared error regression
    n_target: int, optional (default=1)
        Number of distinct outputs. Each will have their own mean and quantiles
        When fitting the model, this should be equal to the number of columns in y_train
    layer_type: str
        Type of recurrent layer
        Options: 'LSTM' (long short-term memory) or 'GRU' (gated recurrent unit)
    dropout: float, optional (default=None)
        Rate parameter for dropout, value in (0,1)
        default is None, which means that no batch dropout is applied
    recurrent_dropout: float or str, optional (default='dropout')
        Parameter for batch normalization, value in (0,1)
        default is 'dropout', which means that recurrent dropout is the same as dropout
    hidden_activation: str (default='relu')
        Activation function for hidden layers, for more explanation:
        https://keras.io/layers/core/
    output_activation: str (default='linear')
        Activation function for output layer, for more explanation:
        https://keras.io/layers/core/
    lr: float (default=0.001)
        Learning rate

    Returns
        keras model

    Examples
    --------
    >>> import pandas as pd
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
    >>> model.fit(X_3d, y, batch_size=32, epochs=5)
    """
    from tensorflow.keras.layers import Input, Dense, Dropout, Activation, GRU, LSTM
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    if len(quantiles) == 0:
        mse_tilted = 'mse'
    else:
        def mse_tilted(y, f):
            loss = keras_joint_mse_tilted_loss(y, f, quantiles, n_target)
            return(loss)
    # Recurrent layer is either LSTM or GRU
    if layer_type == 'LSTM':
        RNN = LSTM
    elif layer_type == 'GRU':
        RNN = GRU
    else:
        raise ValueError('Invalid layer_type, choose "LSTM" or "GRU"')
    if recurrent_dropout == 'dropout':
        recurrent_dropout = dropout
    n_out = n_target * (len(quantiles) + 1)  # one extra for mean regression
    input_layer = Input(input_shape)
    h = input_layer
    for i in range(n_layers):
        # last layer returns last value of sequence:
        return_sequences = (i < n_layers - 1)
        h = RNN(n_neurons,
                activation=hidden_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=return_sequences)(h)
    out = Dense(n_out, activation=output_activation)(h)
    model = Model(inputs=input_layer, outputs=out)
    model.compile(loss=mse_tilted,
                  optimizer=Adam(lr=lr))
    return(model)
