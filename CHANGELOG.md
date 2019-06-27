# Changelog

## Version 1.0.2

Change `decompose_datetime` and `recode_cyclical_features`: the `remove_original` argument has been deprecated and renamed to `remove_categorical`. The original name was wrong, since this parameter never removed the original features, but only the newly created categorical features.

Change `decompose_datetime` and `recode_cyclical_features`: a new parameter `keep_original` has been added. This parameter behaves the same as `BuildRollingFeatures`: it is True by default, but can be set to False to keep only the newly created features.

Add new functions: `keras_tilted_loss`, `keras_rmse`, `get_keras_forecasting_metrics`.

Improve `read_regenradar`: it now allows optional arguments to be passed directly to the lizard API. Unfortunately, as of now, we still don't have access to lizard API documentation, so usefulness of this new feature is limited.

## Version 1.0.1

Change `normalize_timestamps` signature and defaults. No UserWarning was given because the previous version was so broken that it needed to be fixed asap

Change `correct_outside_range`, `correct_below_threshold`, `correct_above_threshold` to accept series instead of a dataframe. The old behavior can be recreated: given `df` with column `TARGET`: The old behavior was `df = correct_outside_range(df, 'TARGET')`, equivalent new code is `df['TARGET'] = correct_outside_range(df['TARGET'])`.

Change `correct_outside_range`, `correct_below_threshold`, `correct_above_threshold` to ignore missing values completely. Previously, missing values were treated as outside the range.

Added new functions: `sam_format_to_wide`, `wide_to_sam_format`, `FunctionTransformerWithNames`

## Version 1.0

First release.