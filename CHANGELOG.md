# Changelog

## Version 2.0.5

### Changes
- `sam.feature_engineering.automatic_rolling_engineering` now has new argument 'onehots'. The argument 'add_time_features' is now removed, as 'cyclicals' and 'onehots' now together make up both timefeatures

## Version 2.0.4

### Changes
- `sam.feature_engineering.decompose_datetime` 'components' argument now support 'secondofday'

## Version 2.0.3

### Changes
- `sam.visualization.quantile_plot` 'score' argument changed to 'title' to enhance generalizability

## Version 2.0.2

### New functions
- New function: `sam.visualization.quantile_plot` function creates an (interactive) plot of SamQuantileMLP output

### Changes
- `sam.feature_engineering.decompose_datetime` now has an new argument 'onehots' that converts time variables to one-hot-encoded
- `sam.feature_engineering.BuildRollingFeatures`: now as an argument 'add_lookback_to_colname'
- `sam.models.SamQuantileMLP`: now has argument 'time_onehots', default time variables adjusted accordingly
- `sam.models.SamQuantileMLP`: now has argument 'y_scaler'

### Bugfixes
- `sam.models.SamQuantileMLP`: setting use_y_as_feature to True would give error if predict ahead was 0.

## Version 2.0.1

### New functions
- New function: `sam.models.create_keras_autoencoder_mlp` function that returns keras MLP for unsupervised anomaly detection
- New function: `sam.models.create_keras_autoencoder_rnn` function that returns keras RNN for unsupervised anomaly detection
- Change `sam.models.create_keras_quantile_mlp`: supports momentum of 1.0 for no batch
normalization. Value of None is still supported.
- Change`sam.models.create_keras_quantile.rnn`: supports lower case layer types 'lstm' and 'gru'

## Version 2.0.0

A lot changed in version 2.0.0. Only changes compared to 1.0.3 are listed here.
For more details about any function, check the documentation.

### New functions

- `sam.preprocessing.RecurrentReshaper` transformer to transform 2d to 3d for Recurrent Neural networks
- `sam.preprocessing.scale_train_test` function that scales train and test set and returns fitted scalers
- `sam.validation.RemoveFlatlines` transformer that finds and removes flatlines from data
- `sam.validation.RemoveExtremeValues` transformer that finds and removes extreme values
- `sam.validation.create_validation_pipe` function that creates sklearn pipeline for data validation
- `sam.preprocessing.make_differenced_target` and `sam.preprocessing.inverse_differenced_target` allow for differencing a timeseries
- `sam.models.SamQuantileMLP` standard model for fitting wide-format timeseries data with an MLP
- `sam.models.create_keras_quantile_rnn` function that returns a keras RNN model that can predict means and quantiles
- Functions for benchmarking a model on some standard data (in sam format): `sam.models.preprocess_data_for_benchmarking`,
  `sam.models.benchmark_model`, `sam.models.plot_score_dicts`, `sam.models.benchmark_wrapper`
- `sam.feature_engineering.AutomaticRollingEngineering` transformer that calculates rolling features in a smart way

### New features

- `sam.data_sources.read_knmi` has an option to use a nearby weather station if the closest weather station contains nans
- `sam.exploration.lag_correlation` now accepts a list as the `lag` parameter
- `sam.visualization.plot_lag_correlation` looks better now
- `sam.recode_cyclical_features` now explicitly requires maximums and provides them for time features
- Added example for SamQuantileMLP at `http://10.2.0.20/sam/examples.html#samquantilemlp-demo`

### Bugfixes

- `sam.preprocessing.sam_format_to_wide` didn't work on pandas 0.23 and older
- `sam.exploration.lag_correlation` did not correctly use the correlation method parameter
- `sam.metrics.keras_tilted_loss` caused the entire package to crash if tensorflow wasn't installed
- `sam.visualization.plot_incident_heatmap` did not correctly set the y-axis
- `sam.feature_engineering.BuildRollingFeatures` threw a deprecationwarning on newer versions of pandas
- General fixes to typos and syntax in the documentation

## Version 1.0.3
Added new functions: `keras_joint_mse_tilted_loss`, `create_keras_quantile_mlp`

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