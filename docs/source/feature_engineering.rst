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

Build timefeatures
-------------------
.. autofunction:: sam.feature_engineering.build_timefeatures

Decompose datetime
------------------
.. autofunction:: sam.feature_engineering.decompose_datetime
