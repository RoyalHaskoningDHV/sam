import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
