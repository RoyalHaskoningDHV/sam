import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class RemoveExtremeValues(BaseEstimator, TransformerMixin):
    """
    This transformer finds extreme values and sets them to nan in a few steps:

    - Estimate upper and lower bounds from the data in the fit method by
      computing median deviation above and below a running median
    - Mark differences outside these bounds as nan in the transform method

    This class can be passed to the plot function (see :ref:`extreme-removal-plot`)
    to create a visualization of the removal procedure.
    It is advisory to take a look at this diagnostic plot to see if your
    `rollingwindow` parameter is sufficiently large to capture slow variations,
    without removing local peaks that might be 'outliers'.

    In addition, the default `madthresh` of 15 is relatively conservative. Less
    strict thresholds can be tried.

    Note that you only pass cols that are suited for extreme value detection.
    For instance, a pump can sometimes be out of operation and so be set to 0.
    This signal is therefore not suited for extreme value detection.

    Note that nans still have to be filled in with a later procedure.

    Note that you should fit this method to the train set!

    Parameters
    ---------
    cols: list of strings
        columns to detect extreme values for
    rollingwindow: int or string
        if number, this amount of values will be used for the rolling window
        if string, should be in pandas timedelta format ('1D'), and data should
        have a datetime index. A sensible value for this depends on your time
        resolution, but you could try values between 200-400.
    madthresh: float
        number of median absolute deviations to use as threshold.

    Examples
    --------
    >>> from sam.validation import RemoveExtremeValues
    >>> from sam.visualization import diagnostic_extreme_removal
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # create some random data
    >>> np.random.seed(10)
    >>> data = np.random.random(size=(1000))
    >>>
    >>> # split in train and test
    >>> train_df = pd.DataFrame()
    >>> train_df['values'] = data[:800]
    >>> test_df = pd.DataFrame()
    >>> test_df['values'] = data[800:]
    >>>
    >>> # with one clear outlier
    >>> train_df.loc[25] *= 10
    >>>
    >>> # now detect extremes
    >>> cols_to_check = ['values']
    >>> REV = RemoveExtremeValues(
    >>>     cols=cols_to_check,
    >>>     rollingwindow=10,
    >>>     madthresh=10)
    >>> train_corrected = REV.fit_transform(train_df)
    >>> fig = diagnostic_extreme_removal(REV, train_df, 'values')
    >>> test_corrected = REV.transform(test_df)
    >>> fig = diagnostic_extreme_removal(REV, test_df, 'values')
    """

    def __init__(self, cols: list, rollingwindow: Union[int, str], madthresh=15):

        self.cols = cols
        self.rollingwindow = rollingwindow
        self.madthresh = madthresh

    def _compute_rolling(self, x: pd.Series):
        r = x.rolling(self.rollingwindow, min_periods=1, center=True).median()
        return r

    def fit(self, data: pd.DataFrame):
        """
        Estimate upper and lower bounds from the data by column by
        computing median deviation above and below a running median by column.
        This method creates the attiburte self.thresh_high and self.thresh_low
        that contain the respective bounds.

        Parameters
        ----------
        data: pd.DataFrame
            with time indices and feature columns
        """

        self.thresh_high = {}
        self.thresh_low = {}
        for c in self.cols:

            # get data
            x = data.loc[:, c]

            # determine differenc
            rolling = self._compute_rolling(x)
            diff = x.values - rolling

            # Define thresholds as number of median absolute deviations.
            # Note that as we calculate the threshold two-sided, they
            # should be computed on signed and not absolute values
            self.thresh_high[c] = np.median(diff[diff > 0]) * self.madthresh
            self.thresh_low[c] = np.median(diff[diff < 0]) * self.madthresh

        return self

    def transform(self, data: pd.DataFrame):
        """
        Sets values that fall outside bounds set in the fit method to nan

        Parameters
        ----------
        data: pd.DataFrame
            with time indices and feature columns

        Returns
        ------
        data_r: pd.DataFrame
            input data with columns marked as nan
        """

        self.rollings, self.invalids, self.diffs = {}, {}, {}
        data_r = data.copy()
        for c in self.cols:

            # get data
            x = data.loc[:, c]

            # determine rolling and diff
            rolling = self._compute_rolling(x)
            diff = x.values - rolling

            # as thresholds are computed in signed way, we can directly compare
            invalids = diff > self.thresh_high[c]
            invalids |= diff < self.thresh_low[c]

            # save some variables to self so they are available for plot
            self.diffs[c] = diff
            self.rollings[c] = rolling
            self.invalids[c] = invalids

            # set false values to nan
            data_r.loc[invalids, c] = np.nan

            # log number of values removed and tresholds used
            logger.info(
                "detected %d " % np.sum(invalids)
                + "extreme values from %s. " % c
                + "using upper threshold of: %.2f " % self.thresh_high[c]
                + "and lower threshold of: %.2f " % self.thresh_low[c]
                + "using madthresh of %d " % self.madthresh
                + "and rollingwindow of %s" % str(self.rollingwindow)
            )

        return data_r
