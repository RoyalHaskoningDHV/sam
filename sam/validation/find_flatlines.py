import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class RemoveFlatlines(BaseEstimator, TransformerMixin):
    """
    Detect flatlines and set to nan. Note that you have to check whether
    signals can contain natural flatliners (such as machines turned off),
    that might not need to be removed.

    Parameters
    ----------
    cols: list of strings (defaults to None)
        columns to apply this method to. If None, will apply to every column.
    window: "auto" or int (default = 1)
        number of previous equal values to consider current value a flatliner.
        so if set to 2, requires that 2 previous values are identical to
        current to set current value to nan.
        If set to "auto", the threshold is derived in the ``fit`` method.
        Based on a train set, the probability of difference being 0
        is estimated. This probability can be used to estimate the
        number of consecutive flatline samples, before the likelihood
        is below the ``pvalue`` parameter
        The maximum acceptable flatline window is derived for each column
        separately, with the same ``pvalue``
    pvalue: float or None (default=None)
        Threshold for likelihood of multiple consecutive flatline samples
        Only used if ``window="auto"``
        Small pvalues lead to a larger threshold, hence less flatlines will be removed
    margin: int (default = 0)
        Maximum absolute difference between consecutive samples to consider them equal.
        Default is 0, which means that consecutive samples must be exactly equal
        to form a flatline.
    backfill: bool (default = True)
        whether to label all within the window, even before the first detected
        data point. This is useful if you want to remove flatlines from the
        beginning of a signal. However, that is not always representative of
        for a real-time application, so one might want to set this to False.

    Examples
    --------
    >>> from sam.validation import RemoveFlatlines
    >>> # create some data
    >>> data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
    >>> # with one clear outlier
    >>> test_df = pd.DataFrame()
    >>> test_df['values'] = data
    >>> # now detect flatlines
    >>> cols_to_check = ['values']
    >>> RF = RemoveFlatlines(
    ...     cols=cols_to_check,
    ...     window=3)
    >>> data_corrected = RF.fit_transform(test_df)
    """

    def __init__(
        self,
        cols: list = None,
        window: Union[int, str] = 1,
        pvalue: float = None,
        margin: float = 0,
        backfill: bool = True,
    ):

        self.cols = cols
        self.window = window
        self.pvalue = pvalue
        self.margin = margin
        self.backfill = backfill

    def fit(self, data: pd.DataFrame):
        """If window size is 'auto', derive thresholds for each column
        Threshold is based on the probability that a sensor value does not change.
        The likelihood of a flatliner of m time steps, is this probability to the power m.
        A threshold such that flatliners with a likelihood below the pvalue are removed.

        Parameters
        ----------
        data: pd.DataFrame
            The dataframe to apply the method to
        """
        self.window_dict = {}
        for col in data.columns:
            if self.window == "auto":
                # estimate p
                p_estimate = (data[col].diff() == 0).mean()
                # compute threshold
                threshold = int(np.ceil(np.log(self.pvalue) / np.log(p_estimate)))
            else:
                threshold = self.window
            self.window_dict[col] = threshold
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data

        Parameters
        ----------
        data: pd.DataFrame
            with index as increasing time and columns as features

        Returns
        -------
        data_r: pd.DataFrame
            with flatlines replaced by nans
        """

        self.invalids = {}
        data_r = data.copy()

        if self.cols is None:
            self.cols = data.columns

        for col in self.cols:

            these_data = data.loc[:, col]

            # check if sequential values are equal
            no_change = (these_data.diff().abs() <= self.margin).astype(int)

            # check if all sequential values are equal within window
            window = self.window_dict[col]
            flatliners = no_change.rolling(window).min().fillna(0)

            # apply backfill if needed: label all points within flatline window
            # as invalid. This requires a forward looking window
            if self.backfill:
                inv_flatliners = flatliners.iloc[::-1]
                inv_flatliners = inv_flatliners.rolling(window + 1, min_periods=1).max()
                flatliners = inv_flatliners.iloc[::-1]

            flatliners = flatliners.astype(bool)

            # save to self for later plot
            self.invalids[col] = flatliners

            logger.info(
                f"detected {np.sum(flatliners)} "
                f"flatline samples in {col} "
                f"with window of {window} "
            )

            # now replace with nans
            data_r.loc[flatliners, col] = np.nan

        return data_r
