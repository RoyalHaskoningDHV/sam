# Changelog

From version 2.5.0 on, we use the [semantic versioning](https://semver.org/) scheme:

Version X.Y.Z stands for:
- X = Major version: if any backwards incompatible changes are introduced to the public API
- Y = Minor version: if new, backwards compatible functionality is introduced to the public API
- Z = Patch version: if only backwards compatible bug fixes are introduced

-------------

## Version 3.1.0

### New features
- New class `sam.models.LassoTimeseriesRegressor` to create a Lasso regression model for time series data incl. quantile predictions.

## Version 3.0.1

No changes, bumped number for release.

## Version 3.0.0

### New features
- New class `sam.feature_engineering.BaseFeatureEngineer` to create a default interface for feature engineering transformers.
- New class `sam.feature_engineering.FeatureEngineer` to make any feature engineering transformer from a function.
- New class `sam.feature_engineering.IdentyEngineer` to make a transformer that only passes data (does nothing). Utility for other features.
- New class `sam.feature_engineering.SimpleFeatureEngineer` for creating time series features: rolling features and time components (one-hot or cyclical)
- Utility functions `sam.models.utils.remove_target_nan` and `sam.models.utils.remove_until_first_value` for removing missings values in training data.

### Changes
- Replaces `SamQuantileMLP` with new `MLPTimeseriesRegressor`, which has more general purpose. Allows to provide any feature engineering transformer / pipeline. Default parameters are changed as well.
- New example notebooks and corresponding datasets for new feature engineering and model classes.
- Renaming name of `SPCRegressor` to `ConstantTimeseriesRegressor` for consistency. Also `SPCTemplate` was renamed to `ConstantTemplate` accordingly.
- Combination of `use_diff_of_y=True` and providing `y_scaler` did not work correctly. Fixed.
- Changed deprecated `lr` to `learning_rate` in `tensorflow.keras.optimizers.Adam`.
- All classes  now support `get_feature_names_out` instead of `get_feature_names`, which is consistent with `scikit-learn>=1.1`.
- Updated documentation and new examples for new feature engineering and model classes. `data/rainbow_beach.parquet` provides a new example dataset.


## Version 2.11.1

### Changes
- Fixed the version info for the Sphinx docs

## Version 2.11.0

### Changes
- Moved to pyproject.toml instead of setup.py to make this package more future proof
- Removed deprecated Azure Devops pipelines

## Version 2.10.3

### Changes
- Added `.readthedocs.yml` and `docs/requirements.txt` to include requirements for readthedocs build.

## Version 2.10.2

### Changes
- Updated `CONTRIBUTING.md` for open source / github contribution guidelines
- Added `black` to requirements and linting pipeline
- All code reformatted with `black` and project configuration

## Version 2.10.1

### Changes
- Revert version changes in `scikit-learn` and `tenforflow` due to compatibility issues

## 2.10.0

### Changes
- `decompose_datetime()` now also accepts a timezone argument. This enables the user to use time features in another timezone. For example: If your input data is in UTC, but you're expecting that human behaviour is also important and the model is applied on the Netherlands, you can add `Europe/Amsterdam` to `decompose_datetime` and it will convert the time from UTC to the correct time, also taking into account daylight savings. This only has an effect on the feature engineering, preprocessing and postprecessing should always happen on UTC dates.
- Fixed mypy errors in decompose_datetime.py
- Updated docstring examples in decompose_datetime.py (they work now)
## Version 2.9.1

### Changes
- MIT License added
- Additional information in `setup.py` and `setup.cfg` for license

## 2.9.0

### Changes
- Updates package dependencies to no longer use a fixed version, but instead a minimum version
- Changed logging submodule to logging_functions to prevent overwriting logging package
- Fixed some mypy errors
- Added fix for SHAP DeepExplainer: https://github.com/slundberg/shap/issues/2189
- Fixed some deprecation warnings

## 2.8.5

### Changes
- `pyproject.toml` provides settings for building package (required for PyPI)
- Additional information in `setup.py` for open source release

## 2.8.4

### Changes
- `predict` method from `sam.models.ConstantTimeseriesRegressor` now accepts kwargs for compatibility. Now, swapping models with `SamQuantileMLP` with `force_monotonic_quantiles` doesn't cause a failure.

## 2.8.3

### Changes
- `sam.models.QuantileMLP` requires `predict_ahead` to be int or list, but always casts to lists. Change to tuples in version 2.6.0, but caused inconsistencies and incorrect if statements.

## 2.8.2

### Changes
- `sam.visualization.sam_quantile_plot` now displays quantiles in 5 decimals, requirement from Aquasuite with larger quantiles.

## 2.8.1

### Changes
- New (optional) parameters for  `sam.validation.RemoveFlatlines`: `backfill` and `margin`
- Simplified `sam.validation.RemoveFlatlines` to use `pandas.DataFrame.rolling` functions

## Version 2.8.0

### Changes
- `SamQuantileMLP.predict` now accepts `force_monotonic_quantiles` to force quantiles to be monotonic using a postprocessing step.

## Version 2.7.0

### Changes
- Added a SPC model to SAM called `ConstantTimeseriesRegressor`, which uses the `SamQuantileRegressor` base class and can be used as a fall back or benchmark model

### Fixes
- `SamQuantileMLP` now accepts Sequence types for some of its init parameters (like quantiles, time_cyclicals etc.) and the default value is changed to tuples to prevent the infamous "Mutable default argument" issue.

## Version 2.6.0

### Changes
- Added a new abstract base class for all SAM models called `SamQuantileRegressor`, that contains some required abstract methods (fit, train, score, dump, load) any subclass needs to implement as well as some default implementations like a standard feature engineer. `SamQuantileMLP` is now a subclass of this new abstract base class, new classes will follow soon. 

## Version 2.5.6

### Changes
- `sam.visualization._evaluate_performance` now checks for nan in both `y_hat` and `y_true`.

## Version 2.5.5

### Changes
- `sam.visualization.performance_evaluation_fixed_predict_ahead` accepts `metric` parameter that indicates what metric to evaluate the performance with: 'R2' or 'MAE' (mean absolute error). Default metric is 'R2'.

## Version 2.5.4

### Changes

- No more bandit linting errors: replace `assert` statements
- Remove faulty try-except-pass constructions

### New features
- Function `sam.utils.contains_nan` and `sam.utils.assert_contains_nan` are added for validation

## Version 2.5.3

### Changes
- Scikit-learn version had to be <0.24.0 for certain features, TODO: update dependencies in the near future
- Updated README, setup.py and CONTRIBUTING in preparation for going open-source.

## Version 2.5.2

### Bugfix
- `LinearQuantileRegression` only contains parameters and pvalues, and data is no longer stored in
the class. This was unwanted.

### New features
- `LinearQuantileRegression` accepts `fit_intercept` parameter, similar to `sklearn.LinearRegression`.

## Version 2.5.1

### Bugfix

- `read_knmi_station_data`
  - Added a with statement to close API connection, which caused errors if used too many times  

## Version 2.5.0

### General changes
- Removed all deprecated functions, see next [subsection](#2.5.0.additional_changes) for details. All deprecated tests have been removed as well.
- All docstrings have been checked and (if needed) updated
- Type hinting in all files
- Linting changes:
   - Changed pipeline linter to flake8
   - Formatted all files in black
   - Split large classes and functions to satisfy a [maximum cyclomatic complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity) of 10
   - Moved inline imports to top of file if the packages were already imported by (any) parent
   - Sorted imports
- Updated the `README.MD` and `CONTRIBUTING.MD` files

### <a id="2.5.0.additional_changes"></a> Additional changes in subpackages
- `sam.data_sources`
    - Deleted deprecated function `sam.data_sources.create_synthetic_timeseries`
- `sam.feature_engineering`
    - Reduced duplicate code in `sam.feature_engineering.automatic_rolling_engineering` and `sam.feature_engineering.decompose_datetime`
    - `sam.feature_engineering.automatic_rolling_engineering`: all dataframe inputs must be linearly increasing in time and have a datetime index, if not an AssertionError is raised
    - Deleted deprecated function `sam.feature_engineering.build_timefeatures`
    - Moved hardcoded data in `sam.feature_engineering.tests.test_automatic_feature_engineering` to separate `test_data` parent folder
- `sam.feature_selection`
    - This subpackage is removed, as it was deprecated
- `sam.models`
    - Reduced complexity of `sam.models.SamQuantileMLP` by adding extra internal methods for large methods
- `sam.preprocessing`
    - Removed merge conflict files `sam.preprocessing\tests\test_scaling.py.orig` and `sam.preprocessing\data_scaling.py.orig`
    - Deleted deprecated function `sam.preprocessing.complete_timestamps`
- `sam.train_models`
    - This subpackage is removed, as it was deprecated
- `sam.utils`
    - Deleted deprecated functions: `sam.utils.MongoWrapper`, `sam.utils.label_dst`, `sam.utils.average_winter_time`, and `sam.utils.unit_to_seconds`
    - Added new function `sam.utils.has_strictly_increasing_index` to validate the datetime index of a dataframe
- `sam.visualization`
    - reduced complexity of `sam.visualization.sam_quantile_plot` by splitting the static and interactive plot in separate functions.

## Version 2.4.0

### New features
- `sam.data_sources.read_knmi_station_data` was added to get KNMI data for a selection of KNMI station numbers
- `sam.data_sources.read_knmi_stations` was added to get all automatic KNMI station meta data

### Bugfixes
- `sam.data_sources.read_knmi` was changed because of a new KNMI API. The package `knmy` does not work anymore. 
- `knmy` is no longer a (optional) dependency (outdated)

## Version 2.3.0

### New features
- `sam.visualization.quantile_plot` accepts `benchmark` parameter that plots the benchmark used to calculate the model performance

### Changes
- `sam.preprocessing.sam_reshape.sam_format_to_wide` now explicitly defines the arguments when calling `pd.pivot_table`
- `sam.metrics.r2_calculation.train_r2` can now use an array as a benchmark, not only a scalar average, for r2 calculation

## Version 2.2.0

### New features
- `sam.visualization.performance_evaluation_fixed_predict_ahead` accepts `train_avg_func` parameter that provides a function to calculate the average of the train set to use for r2 calculation (default=np.nanmean)

### New functions
- Name change: `sam.metrics.train_mean_r2` -> `sam.metrics.r2_calculation` to avoid circular import errors and the file now contains multiple methods
- New function: `sam.metrics.r2_calculation.train_r2` a renamed copy of `sam.metrics.r2_calculation.train_mean_r2` as any average can now be used for r2 calculation

### Changes
- `sam.metrics.train_mean_r2` is now deprecated and calls `sam.metrics.train_r2`

## Version 2.1.0

### New features
- `sam.data_sources.read_knmi` now accepts parameter `preprocessing` to transform data to more scales.

## Version 2.0.22

### New features
- `keras_joint_mae_tilted_loss`: to fit the median in quantile regression (use average_type='median' in SamQuantileMLP)
- `plot_feature_importances`: bar plot of feature importances (e.g. computed in SamQuantileMLP.quantile_feature_importances
- `compute_quantile_ratios`: to check the proportion of data falling beneath certain quantile

## Version 2.0.21

### Bugfixed
- eli5 uses the sklearn.metrics.scorer module, which is gone in 0.24.0, so we need <=0.24.0
- shap does not work with tensorflow 2.4.0 so we need <=2.3.1

## Version 2.0.20

### Bugfixed
- statsmodels is no longer a dependency (dependency introduced in version 2.0.19)

## Version 2.0.19

### New features
- `sam.metrics.tilted_loss`: A tilted loss function that works with numpy / pandas
- `sam.models.LinearQuantileRegression`: sklearn style wrapper for quantile regression using statsmodels

## Version 2.0.18

### Changes
- `sam.models.SamQuantileMLP`: Now stores the input columns (before featurebuilding) which can be accessed by `get_input_cols()`


## Version 2.0.17

### Changes
- `sam.validation.flatline`: Now accepts `window="auto"` option, for which the maximum flatline window is estimated in the `fit` method

## Version 2.0.16

### New functions
- New class: `sam.feature_engineering.SPEITransformer` for computing precipitation and evaporation features

## Version 2.0.15

### Bugfixes
- Fixed failing unit tests by removing tensorflow v1 code
- Fixed QuantileMLP, where the target would stay an integer, which fails with our custom loss functions
- Updated optional dependencies to everything we use
- With the latest pandas version a UTC to string conversio has been fixed. Removed our fix, upped the pandas version
- Updated scikit-learn to at least 0.21, which is required for the iterative imputer

### Development changes
- Added `run-linting.yml`  to run pycodestyle in devops pipelines
- Added `run-unittest.yml` to run pytest in devops pipelines
- Removed `.arcconfig` (old arcanist unit test configuration)
- Removed `.arclint` (old arcanist lint configuration)

## Version 2.0.14

### New functions
- `sam.visualisation.sam_quantile_plot`: Options to set `outlier_window` and `outlier_limit`, to only plot anomalies when at least `outlier_limit` anomalies are counted within the `outlier window`

### Bugfixes
- Bugfix in `sam.metrics.custom_callbacks`

## Version 2.0.11

### Bugfixes
- `sam.models.SamQuantileMLP.score`: if using y_scaler, now scales actual and prediction to equalize score to keras loss

## Version 2.0.10

### New functions
- `sam.models.SamQuantileMLP.quantile_feature_importances`: now has argument sum_time_components that summarizes feature importances for different features generated for a single component (i.e. in onehot encoding).

### Changes
- `sam.featurew_engineering.automatic_rolling_engineering`: `estimator_type` argument can now also be 'bayeslin', which should be used if one hot components are used

### Bugfixes
- `sam.featurew_engineering.automatic_rolling_engineering`: constant features are no longer deleted (broke one hot features)

## Version 2.0.9

### Bugfixes
- `sam.models.SamQuantileMLP`: When using y_scaler, name of rescaled y-series is set correctly.

### Changes
- `sam.models.SamQuantileMLP`: Now accepts a keyword argument `r2_callback_report` to add the new custom r2 callback.

### New functions
- `sam.metrics.custom_callbacks`: Added a custom callback that computes r2 with `sam.metrics.train_mean_r2` for each epoch

## Version 2.0.8

### Bugfixes
- `sam.validation.create_validation_pipe`: the imputation part is now correctly applied only to the `cols` columns in the df
- `sam.metrics.train_mean_r2`: now only adds non-nan values in np.arrays (previously would return nan R2)

## Version 2.0.7

### Changes
- `sam.visualization.quantile_plot`: now accepts custom outliers with 'outlier' argument

### Bugfixes
- `sam.visualization.quantile_plot`: now correctly shifts y_hat with predict_ahead

## Version 2.0.6

### New functions
- New function: `sam.metrics.train_mean_r2` that evaluates r2 based on the train set mean
- New function: `sam.visualization.performance_evaluation_fixed_predict_ahead` that evaluates model performance with certain predict ahead.

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