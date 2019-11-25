import pandas as pd
import numpy as np
from sam.feature_engineering import BuildRollingFeatures


def make_differenced_target(y, lags=1, newcol_prefix=None):
    '''
    Creates a target dataframe by performing differencing on a series

    Given some features, it may be desirable to predict future values of the target.
    In this case, it is often desirable to perform differencing.
    Also, it may be desirable to have either one or multiple targets. To preserve consistency,
    this function returns a dataframe either way.

    This function creates a dataframe with columns 'TARGET_diff_x', where x are the lags,
    and TARGET is the name of the input series.

    Parameters
    ----------
    y: pd.Series
        A series containing the target data. Must be monospaced in time, for the differencing
        to work correctly.
    lags: array-like or int, optional (default=1)
        A list of integers, or a single integer describing what lags should be used to look in the
        future. For example, if this is [1, 2, 3], the output will have three columns, performing
        differencing on 1, 2, and 3 timesteps in the future.
        If this is a list, the output will be a dataframe. If this is a scalar, the output will
        be a series
    newcol_prefix: str, optional (default=None)
        The prefix that the output columns will have. If not given, `y.name` is used instead.

    Returns
    -------
    target: pd.DataFrame or pd.Series
        A target with the same index as y, and columns equal to len(lags)
        The values will be 'future values' of y but differenced.
        If we consider the index to be the 'timestamp', then the index will be the moment the
        prediction is made, not the moment the prediction is about. Therefore, the columns
        will be different future values with different lags.
        Any values that cannot be calculated (because there is no available future value) will be
        set to np.nan.

    Examples
    --------
    >>> df = pd.DataFrame({
    >>>     'X': [18, 19, 20, 21],
    >>>     'y': [10, 20, 50, 100]
    >>> })
    >>> make_differenced_target(df['y'], lags=1)
        y_diff_1
    0 	10.0
    1 	30.0
    2 	50.0
    3 	NaN
    '''
    if newcol_prefix is None:
        newcol_prefix = y.name

    if np.isscalar(lags):
        series_output = True
        lags = [lags]
    else:
        series_output = False

    for lag in lags:
        if lag < 1:
            raise ValueError("All lags must be larger than 0")
        if lag % 1 != 0:
            raise ValueError("All lags must be integers")

    # When looking at the future, the diffs should negative, so we need to multiply by -1
    # Also, lagging to the future means negative lags
    result = pd.concat([-1 * y.diff(-1 * lag) for lag in lags], axis=1)

    names = ['{}_diff_{}'.format(newcol_prefix, lag) for lag in lags]
    result.columns = names
    if series_output:
        result = result.iloc[:, 0]

    return result


def inverse_differenced_target(predictions, y):
    '''
    Inverses differencing by adding the current values to the prediction.

    This function will take differenced target(s) and the current values, and return the actual
    target(s). Can be used to convert predictions from a differenced model to real predictions.

    predictions and y must be joined on index. Any indexes that only appear in predictions, or
    only appear in y, will also appear in the output, with nans inserted.

    Parameters
    ----------
    predictions: pd.DataFrame
        Dataframe containing differenced values.
    y: pd.Series
        The actual values in the present

    Returns
    -------
    actual: pd.DataFrame
        Dataframe containing un-differenced values, created by adding predictions to y on index.
        The index of this output will be the union of the indexes of predictions and y.
        The columns refer to the values/predictions made at a single point in time.
        For example, if the index is '18:00', and the predictions are made on differencing 1 hour,
        2 hour and 3 hours, then one row will contain the predictions made at 18:00,
        predicting what the target will be at 19:00, 20:00 and 21:00

    Examples
    --------
    >>> df = pd.DataFrame({
    >>>     'X': [18, 19, 20, 21],
    >>>     'y': [10, 20, 50, 100]
    >>> })
    >>> target = make_differenced_target(df['y'], lags=1)
    >>> inverse_differenced_target(target, df['y'])
      y_diff_1
    0 20.0
    1 50.0
    2 100.0
    3 NaN

    >>> prediction = pd.DataFrame({
        'pred_diff_1': [15, 25, 34, np.nan],
        'pred_diff_2': [40, 55, np.nan, np.nan]
    })
    >>> inverse_differenced_target(prediction, df['y'])
      pred_diff_1   pred_diff_2
    0 25.0          60.0
    1 45.0          105.0
    2 84.0          NaN
    3 NaN           NaN
    >>> # This means that at timestep 0, we predict that the next two values will be 25 and 60
    >>> # At timestep 1, we predict the next two values will be 45 and 105
    >>> # At timestep 2, we predict the next two values will be 84 and unknown, etcetera.
    '''
    return predictions.add(y, axis=0)