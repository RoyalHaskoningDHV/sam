.. _feature_engineering:

===================
Feature Engineering
===================

This is the documentation for feature engineering. Please see [Features](general_documents/feature_extraction.md) for further details on which features to use.

Rolling Features
----------------
.. autoclass:: sam.feature_engineering.BuildRollingFeatures
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: sam.feature_engineering.lag_range_column

Decompose datetime
------------------
.. autofunction:: sam.feature_engineering.decompose_datetime

Cyclical features
-----------------
.. autofunction:: sam.feature_engineering.recode_cyclical_features