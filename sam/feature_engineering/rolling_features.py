import logging
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from sam.logging_functions import log_dataframe_characteristics, log_new_columns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


def _nfft_helper(
    series: pd.Series,
    nfft: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Helper function to apply nfft to series.

    Parameters
    ----------
    series : pd.Series
        series to apply nfft on.
    nfft : function
        nfft function.

    Returns
    -------
    np.ndarray
        result after applying nfft on series.
    """
    # series should be even sample, cutoff one if it is not
    if series.size % 2 != 0:
        series = series.iloc[1:]
    if series.size <= 5:  # too short
        return np.array([])

    # first convert time index to nanoseconds since epoch
    # If a non-pandas timestamp index is used, these units aren't nanoseconds
    # However, since we normalize to [-0.5, 0.5] anyway, it doesn't matter
    time = np.array(series.index.astype("int64"))
    # then make first value 0
    time -= np.min(time)
    # normalize to run from -0.5 to 0.5
    time = time / np.max(time) - 0.5

    # now do fft on normalized values
    f = nfft(time, series.values - np.mean(series.values))
    # flip fft spectrum so it matches the numpy spectrum
    ff = np.fft.fftshift(f)
    # Only take first half since that is useful. See documentation inside
    # multicol_output.Helper.__init__ for better explanation
    return np.abs(ff)[1 : len(ff) // 2]


def multicol_output(
    arr: np.ndarray,
    n: int,
    func: Callable[[np.ndarray], np.ndarray],
    fourier: bool = False,
    time_window: str = None,
) -> pd.DataFrame:
    """
    Generic function to compute multiple columns
    func is a function that takes in a numpy array, and outputs a numpy array of the same length
    The numpy array will be a window: e.g. if n=3, func will get a window of 3 points, and output
    3 values. Then, those 3 values will be converted to columns in a dataframe.
    For fourier, an additional column selection is done, so less than 3 columns would be returned
    In that case, the option 'fourier' needs to be set to True

    For nfft, we need not only a numpy array of values, but also a DatetimeIndex. Additionally, we
    have a time_window instead of an integer window. This time_window is for example '100min'.
    If time_window is set, this function will assume that arr has a DatetimeIndex. n is still used
    as the output size.

    Parameters
    ----------
    arr : np.ndarray
        input numpy array to generate multipole columns from.
    n : int
        window size in number of datapoints.
    func : function
        aggregate function to apply to the rolling series.
    fourier : bool, optional
        enable fourier transformation, by default False.
    time_window : str, optional
        size of the time window in pandas timedelta format, by default None.

    Returns
    -------
    pd.DataFrame
        Result after applying a rolling function with size `n` and aggregate function `func`.
    """

    class Helper:
        # https://stackoverflow.com/a/39064656
        def __init__(self, nrow, n):
            if fourier:
                """we are only interested in these coefficients, since the rest is redundant
                See https://en.wikipedia.org/wiki/Discrete_Fourier_transform
                It follows from the definition that when k = 0, the result is simply the sum
                It also follows that the definition for n-k is the same as k, except inverted
                Since we take the absolute value, this inversion is removed, so the result is
                identical. Therefore, we only want the values from k = 1 to k = n//2
                """
                self.useful_coeffs = range(1, n // 2 + 1)
            else:
                self.useful_coeffs = range(0, n)
            ncol = len(self.useful_coeffs)
            self.series = np.full((nrow, ncol), np.nan)
            if time_window is None:
                self.calls = n - 1
            else:
                self.calls = 0
            self.n = n

        def calc_func(self, vector):
            if len(vector) < self.n and time_window is None:
                # We are still at the beginning of the dataframe, nothing to do
                return np.nan
            values = func(vector)
            # If function did not return enough values, pad with zeros
            values = np.pad(values, (0, max(0, self.n - values.size)), "constant")
            values = values[self.useful_coeffs]
            self.series[self.calls, :] = values
            self.calls = self.calls + 1
            return np.nan  # return something to make Rolling apply not error

    helper = Helper(len(arr), n)
    if time_window is None:
        arr.rolling(n, min_periods=0).apply(helper.calc_func, raw=True)
    else:
        # For time-window, we need raw=False because 'func' may need
        # The DatetimeIndex. Even though raw=False is slower.
        arr.rolling(time_window).apply(helper.calc_func, raw=False)
    return pd.DataFrame(helper.series)


class BuildRollingFeatures(BaseEstimator, TransformerMixin):
    """Applies some rolling function to a pandas dataframe

    This class provides a stateless transformer that applies to each column in a dataframe.
    It works by applying a certain rolling function to each column individually, with a
    window size. The rolling function is given by rolling_type, for example 'mean',
    'median', 'sum', etcetera.

    An important note is that this transformer assumes that the data is sorted by time already!
    So if the input dataframe is not sorted by time (in ascending order), the results will be
    completely wrong.

    A note about the way the output is rolled: in case of 'lag' and 'diff', the output will
    always be lagged, even if lookback is 0. This is because these functions inherently look
    at a previous cell, regardless of what the lookback is. All other functions will start
    by looking at the current cell if lookback is 0. (and will also look at previous cells
    if `window_size` is greater than 1)

    'ewm' looks at `window_size` a bit different: instead of a discrete number of points to
    look at, 'ewm' needs a parameter alpha between 0 and 1 instead.

    Parameters
    ----------
    window_size: array-like, shape = (n_outputs, ), optional (default=None)
        vector of values to shift. Ignored when rolling_type is ewm
        if integer, the window size is fixed, and the timestamps are assumed to be uniform.
        If string of timeoffset (for example '1H'), the input dataframe must have a DatetimeIndex.
        timeoffset is not supported for rolling_type 'lag', 'fourier', 'ewm', 'diff'!
    lookback: number type, optional (default=1)
        the features that are built will be shifted by this value
        If more than 0, this prevents leakage
    rolling_type: string, optional (default="mean")
        The rolling function. Must be one of: 'median', 'skew', 'kurt', 'max', 'std', 'lag',
        'mean', 'diff', 'sum', 'var', 'min', 'numpos', 'ewm', 'fourier', 'cwt', 'trimmean'
    deviation: str, optional (default=None)
        one of ['subtract', 'divide']. If this option is set, the resulting column will either
        have the original column subtracted, or will be divided by the original column. If None,
        just return the resulting column. This option is not allowed when rolling_type is 'cwt'
        or 'fourier', but it is allowed with all other rolling_types.
    alpha: numeric, optional (default=0.5)
        if rolling_type is 'ewm', this is the parameter alpha used for weighing the samples.
        The current sample weighs alpha, the previous sample weighs alpha*(1-alpha), the
        sample before that weighs alpha*(1-alpha)^2, etcetera. Must be in (0, 1]
    width: numeric, optional (default=1)
        if rolling_type is 'cwt', the wavelet transform uses a ricker signal. This parameter
        defines the width of that signal
    nfft_ncol: numeric, optional (default=10)
        if rolling_type is 'nfft', there needs to be a fixed number of columns as output, since
        this is unknown a-priori. This means the number of output-columns will be fixed. If
        nfft has more outputs, and additional outputs are discarded. If nfft has less outputs,
        the rest of the columns are right-padded with 0.
    proportiontocut: numeric, optional (default=0.1)
        if rolling_type is 'trimmean', this is the parameter used to trim values on both tails
        of the distribution. Must be in [0, 0.5). Value 0 results in the mean, close to 0.5
        approaches the median.
    keep_original: boolean, optional (default=True)
        if the original columns should be kept or discarded
        True by default, which means the new columns are added to the old ones
    timecol: str, optional (default=None)
        Optional, the column to set as the index during transform. The index is restored before
        returning. This is only useful when using a timeoffset for window_size, since that needs
        a datetimeindex. So this column can specify a time column. This column will not be
        feature-engineered, and will never be returned in the output!
    add_lookback_to_colname: bool, optional (default=False)
        Whether to add lookback to the newly generated column names.
        if False, column names will be like: DEBIET#mean_2
        if True, column names will be like: DEBIET#mean_2_lookback_0

    Examples
    --------
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> import pandas as pd
    >>> df = pd.DataFrame({'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...                    'DEBIET': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3]})
    >>>
    >>> BuildRollingFeatures(rolling_type='lag', window_size = [0,1,4], \\
    ...                      lookback=0, keep_original=False).fit_transform(df)
        RAIN#lag_0  DEBIET#lag_0  ...  RAIN#lag_4  DEBIET#lag_4
    0          0.1             1  ...         NaN           NaN
    1          0.2             2  ...         NaN           NaN
    2          0.0             3  ...         NaN           NaN
    3          0.6             4  ...         NaN           NaN
    4          0.1             5  ...         0.1           1.0
    5          0.0             5  ...         0.2           2.0
    6          0.0             4  ...         0.0           3.0
    7          0.0             3  ...         0.6           4.0
    8          0.0             2  ...         0.1           5.0
    9          0.0             4  ...         0.0           5.0
    10         0.0             2  ...         0.0           4.0
    11         0.0             3  ...         0.0           3.0
    <BLANKLINE>
    [12 rows x 6 columns]
    """

    def __init__(
        self,
        rolling_type: str = "mean",
        lookback: int = 1,
        window_size: Optional[str] = None,
        deviation: Optional[str] = None,
        alpha: float = 0.5,
        width: int = 1,
        nfft_ncol: int = 10,
        proportiontocut: float = 0.1,
        timecol: Optional[str] = None,
        keep_original: bool = True,
        add_lookback_to_colname: bool = False,
    ):
        self.window_size = window_size
        self.lookback = lookback
        self.rolling_type = rolling_type
        self.deviation = deviation
        self.alpha = alpha
        self.width = width
        self.nfft_ncol = nfft_ncol
        self.proportiontocut = proportiontocut
        self.keep_original = keep_original
        self.timecol = timecol
        self.add_lookback_to_colname = add_lookback_to_colname
        logger.debug(
            "Initialized rolling generator. rolling_type={}, lookback={}, "
            "window_size={}, deviation={}, alpha={}, proportiontocut={}, width={}, "
            "keep_original={}, timecol={}".format(
                rolling_type,
                lookback,
                window_size,
                deviation,
                alpha,
                proportiontocut,
                width,
                keep_original,
                timecol,
            )
        )

    def _validate_params(self):
        """apply various checks to the inputs of the __init__ function
        throw value error or type error based on the result
        """

        self._validate_lookback()
        self._validate_width()
        self._validate_alpha()
        self._validate_proportiontocut()

        if not isinstance(self.rolling_type, str):
            raise TypeError("rolling_type must be a string")

        if not isinstance(self.keep_original, bool):
            raise TypeError("keep_original must be a boolean")

        if int(self.nfft_ncol) != self.nfft_ncol:
            raise ValueError("nfft_ncol must be an integer!")

        if self.deviation not in [None, "subtract", "divide"]:
            raise ValueError("Deviation must be one of [None, 'subtract', 'divide']")

        if self.window_size is None and self.rolling_type != "ewm":
            raise ValueError("Window_size must not be None, unless rolling_type is ewm")

        if self.deviation is not None and self.rolling_type in ["fourier", "cwt"]:
            raise ValueError("Deviation cannot be used together with {}".format(self.rolling_type))

    def _validate_lookback(self):
        if not np.isscalar(self.lookback):
            raise TypeError("lookback must be a scalar")
        if self.lookback < 0:
            raise ValueError("lookback cannot be negative!")

    def _validate_width(self):
        if not np.isscalar(self.width):
            raise TypeError("width must be a scalar")
        if self.width <= 0:
            raise ValueError("width must be positive")

    def _validate_alpha(self):
        if not np.isscalar(self.alpha):
            raise TypeError("alpha must be a scalar")
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")

    def _validate_proportiontocut(self):
        if not np.isscalar(self.proportiontocut):
            raise TypeError("proportiontocut must be a scalar")
        if self.proportiontocut >= 0.5 or self.proportiontocut < 0:
            raise ValueError("proportiontocut must be in [0, 0.5)")

    def _get_rolling_fun(
        self,
        rolling_type: str = "mean",
    ) -> Callable[[Union[pd.Series, np.ndarray], Union[int, None]], Union[pd.Series, np.ndarray]]:
        """Given a function name as a string, creates a function that
        applies that rolling function

        Parameters
        ----------
        rolling_type : string, default="mean"
            the description of the rolling function. must be one of lag, sum, mean,
            median, trimmean, var, std, max, min, skew, kurt, diff, numpos, ewm, fourier, cwt

        Returns
        -------
        rolling_function : function
            function with two inputs: a pandas series and an integer. Will apply
            some rolling function to the series, with window size of the integer.
            Alternatively, in case of fourier/cwt, a function with one input:
            a numpy array. Will output another numpy array.
        """
        if self.rolling_type == "cwt":
            from scipy import signal  # Only needed for this rolling type
        if self.rolling_type == "nfft":
            from nfft import nfft
        if self.rolling_type == "trimmean":
            from scipy.stats import trim_mean

        rolling_functions = {
            "lag": lambda arr, n: arr.shift(n),
            "sum": lambda arr, n: arr.rolling(n).sum(),
            "mean": lambda arr, n: arr.rolling(n).mean(),
            "trimmean": lambda arr, n: arr.rolling(n).apply(
                lambda w: trim_mean(w, self.proportiontocut), raw=True
            ),
            "median": lambda arr, n: arr.rolling(n).median(),
            "var": lambda arr, n: arr.rolling(n).var(),
            "std": lambda arr, n: arr.rolling(n).std(),
            "max": lambda arr, n: arr.rolling(n).max(),
            "min": lambda arr, n: arr.rolling(n).min(),
            "skew": lambda arr, n: arr.rolling(n).skew(),
            "kurt": lambda arr, n: arr.rolling(n).kurt(),
            "diff": lambda arr, n: arr.diff(n),
            "numpos": lambda arr, n: arr.gt(0).rolling(n).sum(),
            "ewm": lambda arr, n: arr.ewm(alpha=self.alpha).mean(),
            # These two have different signature because they are called by multicol_output
            "fourier": lambda vector: np.absolute(np.fft.fft(vector)),
            "cwt": lambda vector: signal.cwt(vector, signal.ricker, [self.width])[0],
            "nfft": lambda series: _nfft_helper(series, nfft),
        }

        if rolling_type not in rolling_functions:
            raise ValueError(
                "The rolling_type is %s, which is not an available function" % rolling_type
            )

        return rolling_functions[rolling_type]

    def _apply_deviation(
        self, arr: np.ndarray, original: np.ndarray, deviation: str
    ) -> np.ndarray:
        """Helper function to apply deviation during the transform"""
        if deviation is None:
            return arr
        if deviation == "subtract":
            return arr - original
        if deviation == "divide":
            # Pandas will insert inf when dividing by 0
            return arr / original

    def _generate_and_add_new_features(
        self, X: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies rolling functions to pandas dataframe `X` and concatenates result to `result`.

        Parameters:
        ----------
        X: pandas dataframe
           the pandas dataframe that you want to apply rolling functions on
        result: pandas dataframe
           the pandas dataframe that you want to add the new features to

        Returns
        -------
        pandas dataframe, shape = `(n_rows, n_features * (n_outputs + 1))`
            the pandas dataframe, appended with the new feature columns
        """

        if self.rolling_type in ["fourier", "cwt", "nfft"]:
            for window_size, suffix in zip(self.window_size_, self.suffix_):
                # If rolling type is nfft, the time_window needs to be set
                if self.rolling_type == "nfft":
                    window_size, time_window = self.nfft_ncol, window_size
                else:
                    time_window = None

                for column in X.columns:
                    new_features = multicol_output(
                        X[column],
                        window_size,
                        self.rolling_fun_,
                        self.rolling_type == "fourier",
                        time_window=time_window,
                    ).shift(self.lookback)
                    # Fourier has less columns
                    if self.rolling_type == "fourier":
                        useful_coeffs = range(1, window_size // 2 + 1)
                    else:
                        useful_coeffs = range(0, window_size)
                    col_prefix = "#".join([str(column), suffix])
                    new_features.columns = ["_".join([col_prefix, str(j)]) for j in useful_coeffs]
                    new_features = new_features.set_index(X.index)
                    result = pd.concat([result, new_features], axis=1)
        else:
            for window_size, suffix in zip(self.window_size_, self.suffix_):
                new_features = X.apply(
                    lambda arr: self._apply_deviation(
                        self.rolling_fun_(arr, window_size).shift(self.lookback),
                        arr,
                        self.deviation,
                    ),
                    raw=False,
                )
                new_features.columns = [
                    "#".join([str(col), suffix]) for col in new_features.columns
                ]
                result = pd.concat([result, new_features], axis=1)

        return result

    def fit(self, X: Any = None, y: Any = None):
        """Calculates window_size and feature function

        Parameters
        ----------
        X: optional, is ignored
        y: optional, is ignored
        """

        self._validate_params()

        if self.rolling_type == "ewm":
            # ewm needs no integer window_size
            self.window_size_ = "ewm"
            self.suffix_ = ["ewm_" + str(self.alpha)]
        else:
            self.window_size_ = self.window_size
            # Singleton window_size is also allowed
            if np.isscalar(self.window_size_):
                self.window_size_ = [self.window_size_]
            self.suffix_ = [
                self.rolling_type + "_" + str(window_size) for window_size in self.window_size_
            ]
            if self.add_lookback_to_colname:
                self.suffix_ = [s + "_lookback_" + str(self.lookback) for s in self.suffix_]
        self.rolling_fun_ = self._get_rolling_fun(self.rolling_type)
        logger.debug(
            "Done fitting transformer. window size: {}, suffix: {}".format(
                self.window_size_, self.suffix_
            )
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms pandas dataframe `X` to apply rolling function

        Parameters
        ----------
        X: pandas dataframe, shape = `(n_rows, n_features)`
           the pandas dataframe that you want to apply rolling functions on

        Returns
        -------
        result: pandas dataframe, shape = `(n_rows, n_features * (n_outputs + 1))`
            the pandas dataframe, appended with the new columns
        """

        check_is_fitted(self, "window_size_")
        if self.keep_original:
            result = X.copy()
        else:
            result = pd.DataFrame(index=X.index)

        if self.timecol is not None:
            # Set DatetimeIndex on the intermediate result
            index_backup = X.index.copy()
            new_index = pd.DatetimeIndex(X[self.timecol].values)
            X = X.set_index(new_index).drop(self.timecol, axis=1)
            if self.keep_original:
                result = result.set_index(new_index).drop(self.timecol, axis=1)
            else:
                result = pd.DataFrame(index=new_index)

        result = self._generate_and_add_new_features(X, result)

        self._feature_names = list(result.columns.values)
        if self.timecol is not None:
            result = result.set_index(index_backup)
        log_new_columns(result, X)
        log_dataframe_characteristics(result, logging.DEBUG)  # Log types as well

        return result

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns feature names for the outcome of the last transform call.
        """
        check_is_fitted(self, "_feature_names")

        return self._feature_names
