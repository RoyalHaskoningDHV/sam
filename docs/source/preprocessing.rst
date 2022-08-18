.. _preprocessing:

=============
Preprocessing
=============

This is the documentation for preprocessing functions.

Clipping data
-------------
.. autoclass:: sam.preprocessing.ClipTransformer
    :members:
    :undoc-members:
    :show-inheritance:

Normalize timestamps
--------------------
.. warning::
   If your each timestamp in your data appears only once (for example, if your data is in wide format), you should almost certainly not use this function.
   Instead, consider using `pd.resample <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html>`_ instead.
   This pandas function does the same as this function, but is more stable, more performant, and has more options.
   The only time in which you should consider using this function, is if the same timestamp can occur multiple times. (for example, if your data is in long format)
.. autofunction:: sam.preprocessing.normalize_timestamps

Correct extremes
----------------
.. autofunction:: sam.preprocessing.correct_outside_range
.. autofunction:: sam.preprocessing.correct_above_threshold
.. autofunction:: sam.preprocessing.correct_below_threshold

Time-specific preprocessing
---------------------------
.. autofunction:: sam.preprocessing.average_winter_time
.. autofunction:: sam.preprocessing.label_dst

.. _sam-format-reshaping-functions:

SAM-format Reshaping
--------------------
.. autofunction:: sam.preprocessing.sam_format_to_wide
.. autofunction:: sam.preprocessing.wide_to_sam_format

Recurrent Features Reshaping
----------------------------
.. autofunction:: sam.preprocessing.RecurrentReshaper

Differencing
------------
.. autofunction:: sam.preprocessing.make_differenced_target
.. autofunction:: sam.preprocessing.inverse_differenced_target
