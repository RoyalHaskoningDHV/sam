.. _preprocessing:

=============
Preprocessing
=============

This is the documentation for preprocessing functions.

Normalize timestamps
-------------------
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

SAM-format Reshaping
--------------------
.. autofunction:: sam.preprocessing.sam_format_to_wide
.. autofunction:: sam.preprocessing.wide_to_sam_format