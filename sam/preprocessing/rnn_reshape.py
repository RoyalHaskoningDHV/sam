import logging

import numpy as np
import pandas as pd
from sam.feature_engineering import BuildRollingFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class RecurrentReshaper(BaseEstimator, TransformerMixin):
    """Reshapes a two-dimensional feature table into a three dimensional table
    sliding window table, usable for recurrent neural networks

    An important note is that this transformer assumes that the data is sorted by time already!
    So if the input dataframe is not sorted by time (in ascending order), the results will be
    completely wrong.

    Given an input array with shape `(n_samples, n_features)`, output array is of shape
    `(n_samples, lookback, n_features)`

    Parameters
    ----------
    window : integer
        Number of rows to look back
    lookback : integer (default=0)
        the features that are built will be shifted by this value.
        If target is in `X`, `lookback` should be greater than 0 to avoid leakage.
    remove_leading_nan : boolean
        Whether leading nans should be removed.
        Leading nans arise because there is no history for first samples

    Examples
    --------
    >>> from sam.data_sources import read_knmi
    >>> from sam.preprocessing import RecurrentReshaper
    >>> X = read_knmi('2018-01-01 00:00:00', '2018-01-08 00:00:00').set_index('TIME')
    >>> reshaper = RecurrentReshaper(window=7)
    >>> X3D = reshaper.fit_transform(X)
    """

    def _validate_params(self):
        """apply various checks to the inputs of the __init__ function
        throw value error or type error based on the result
        """
        if self.lookback < 0:
            raise ValueError("lookback cannot be negative!")
        if not np.isscalar(self.window):
            raise ValueError("window should be a scalar")

    def __init__(self, window, lookback=1, remove_leading_nan=False):
        self.window = window
        self.lookback = lookback
        self.remove_leading_nan = remove_leading_nan
        self.window_range = range(window - 1, -1, -1)
        self.start = window + lookback - 1
        logger.debug(
            "Initialized reshaping generator. window={}, lookback={}, "
            "remove_leading_nan={}".format(window, lookback, remove_leading_nan)
        )
        if self.remove_leading_nan:
            logger.warning(
                "remove_leading_nan is True and can lead to "
                "X and y of unequal size! Be sure that this is what you want."
            )

    def fit(self, X=None, y=None):
        """Calculates n_features

        Parameters
        ----------
        X : Two dimensional dfeature table
        y : optional, is ignored
        """
        self._validate_params()
        self.n_features_ = X.shape[1]
        self.lag_transformer_ = BuildRollingFeatures(
            rolling_type="lag",
            window_size=self.window_range,
            lookback=self.lookback,
            keep_original=False,
        )
        self.lag_transformer_.fit(X)
        return self

    def transform(self, X):
        """Transforms feature table X to apply rolling function and reshaping

        Parameters
        ----------
        X : pandas Dataframe or numpy array, shape = `(n_rows, n_features)`
            feature table (so no ID, TYPE, TIME columns) to transform to three dimensional

        Returns
        -------
        X_new : numpy array
            A three dimensional numpy array, moving windows over the features table `X`
        """
        check_is_fitted(self, "n_features_")
        # X needs to be pandas dataframe to use BuildRollingFeatures
        X = pd.DataFrame(X)
        n_samples = X.shape[0]
        X_lags = self.lag_transformer_.transform(X)
        X_new = np.reshape(X_lags.values, (n_samples, self.window, self.n_features_))
        # remove first elements that have no full history
        if self.remove_leading_nan:
            X_new = X_new[self.start :, :, :]
        return X_new
