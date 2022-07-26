.. _models:

=============
Models
=============

This is the documentation for modeling functions.

Linear Quantile Regression
---------------------------
.. autoclass:: sam.models.LinearQuantileRegression
    :members:
    :undoc-members:
    :show-inheritance:

Keras templates
---------------------------
.. _create-keras-quantile-mlp:
.. autofunction:: sam.models.create_keras_quantile_mlp
.. autofunction:: sam.models.create_keras_quantile_rnn
.. autofunction:: sam.models.create_keras_autoencoder_mlp
.. autofunction:: sam.models.create_keras_autoencoder_rnn

Statistical process control
---------------------------

.. autoclass:: sam.models.ConstantTemplate
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: sam.models.ConstantTimeseriesRegressor
    :members:
    :undoc-members:
    :show-inheritance:

Multilayer Perceptron (MLP)
---------------------------
.. autoclass:: sam.models.MLPTimeseriesRegressor
    :members:
    :undoc-members:
    :show-inheritance:

Benchmarking
----------------------------
.. autofunction:: sam.models.preprocess_data_for_benchmarking
.. autofunction:: sam.models.benchmark_model
.. autofunction:: sam.models.plot_score_dicts
.. autofunction:: sam.models.benchmark_wrapper
