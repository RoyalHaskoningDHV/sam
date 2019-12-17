.. _models:

=============
Models
=============

This is the documentation for modeling functions.

Keras templates
---------------------------
.. _create-keras-quantile-mlp:
.. autofunction:: sam.models.create_keras_quantile_mlp
.. autofunction:: sam.models.create_keras_quantile_rnn

SAM Quantile MLP Model
---------------------------
.. autoclass:: sam.models.SamQuantileMLP
    :members:
    :undoc-members:
    :show-inheritance:

Benchmarking
----------------------------
.. autofunction:: sam.models.preprocess_data_for_benchmarking
.. autofunction:: sam.models.benchmark_model
.. autofunction:: sam.models.plot_score_dicts
.. autofunction:: sam.models.benchmark_wrapper
