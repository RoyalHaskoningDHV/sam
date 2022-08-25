import logging
from typing import Union

import numpy as np
import pandas as pd
from sam.utils import add_future_warning
from sam.validation import BaseValidator

logger = logging.getLogger(__name__)


class FlatlineValidator(BaseValidator):
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
        Small pvalues lead to a larger threshold, hence less flatlines will be
        removed
    margin: int (default = 0)
        Maximum absolute difference within window to consider them equal.
        Default is 0, which means that all samples within used window must be
        exactly equal to form a flatline.
    backfill: bool (default = True)
        whether to label all within the window, even before the first detected
        data point. This is useful if you want to remove flatlines from the
        beginning of a signal. However, that is not always representative of
        for a real-time application, so one might want to set this to False.

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.validation import FlatlineValidator
    >>> # create some data
    >>> data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
    >>> # with one clear outlier
    >>> test_df = pd.DataFrame()
    >>> test_df['values'] = data
    >>> # now detect flatlines
    >>> cols_to_check = ['values']
    >>> RF = FlatlineValidator(
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
        super().__init__()
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

    def _validate_column(
        self,
        data: pd.Series,
        window: Union[int, str],
    ) -> pd.Series:
        """
        Validates a single column against the fitted dataframe
        """
        data = data.copy()

        # check if values within range are within margin
        flatliners = (
            ((data.rolling(window).max() - data.rolling(window).min()) <= self.margin)
            .astype(int)
            .fillna(0)
        )

        # apply backfill if needed: label all points within flatline window
        # as invalid. This requires a forward looking window
        if self.backfill:
            inv_flatliners = flatliners.iloc[::-1]
            inv_flatliners = inv_flatliners.rolling(window, min_periods=1).max()
            flatliners = inv_flatliners.iloc[::-1]

        flatliners = flatliners.astype(bool)

        return flatliners

    def validate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the dataframe against the fitted dataframe. Returns a boolean
        dataframe where True indicates an invalid value.

        Parameters
        ----------
        X: pd.DataFrame
            Input dataframe to validate
        """
        invalid_data = pd.DataFrame(
            data=np.zeros_like(X.values).astype(bool),
            index=X.index,
            columns=X.columns,
        )

        for col in self.cols:
            window = self.window_dict[col]
            invalid_data[col] = self._validate_column(X[col], window)

            logger.info(
                f"detected {np.sum(invalid_data[col])} "
                f"flatline samples in {col} "
                f"with window of {window} "
            )

        return invalid_data


class RemoveFlatlines(FlatlineValidator):
    @add_future_warning("RemoveFlatlines is deprecated, use FlatlineValidator instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
