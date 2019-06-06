# Changelog

## Version 1.0.1

Change `normalize_timestamps` signature and defaults. No UserWarning was given because the previous version was so broken that it needed to be fixed asap

Change `correct_outside_range`, `correct_below_threshold`, `correct_above_threshold` to accept series instead of a dataframe. The old behavior can be recreated: given `df` with column `TARGET`: The old behavior was `df = correct_outside_range(df, 'TARGET')`, equivalent new code is `df['TARGET'] = correct_outside_range(df['TARGET'])`.

Change `correct_outside_range`, `correct_below_threshold`, `correct_above_threshold` to ignore missing values completely. Previously, missing values were treated as outside the range.

Added new functions: sam_format_to_wide, wide_to_sam_format, FunctionTransformerWithNames

## Version 1.0

First release.