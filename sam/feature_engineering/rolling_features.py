import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy import signal  # For cwt
from sam.utils.time import unit_to_seconds
from sam.logging import log_dataframe_characteristics, log_new_columns
import logging
logger = logging.getLogger(__name__)


def multicol_output(arr, n, func, fourier=False):
    """
    Generic function to compute multiple columns
    func is a function that takes in a numpy array, and outputs a numpy array of the same length
    The numpy array will be a window: e.g. if n=3, func will get a window of 3 points, and output
    3 values. Then, those 3 values will be converted to columns in a dataframe.
    For fourier, an additional column selection is done, so less than 3 columns would be returned
    In that case, the option 'fourier' needs to be set to True
    """
    class Helper:
        # https://stackoverflow.com/a/39064656
        def __init__(self, nrow, n):
            if fourier:
                """ we are only interested in these coefficients, since the rest is redundant
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
            self.calls = n - 1
            self.n = n

        def calc_func(self, vector):
            if len(vector) < self.n:
                # We are still at the beginning of the dataframe, nothing to do
                return np.nan
            values = func(vector)[self.useful_coeffs]
            self.series[self.calls, :] = values
            self.calls = self.calls + 1
            return np.nan  # return something to make Rolling apply not error

    helper = Helper(len(arr), n)
    arr.rolling(n, min_periods=0).apply(helper.calc_func, raw=True)
    return pd.DataFrame(helper.series)


class BuildRollingFeatures(BaseEstimator, TransformerMixin):
    """Applies some rolling function to a pandas dataframe

    This class provides a stateless transformer that applies to each column in a dataframe.
    It works by applying a certain rolling function to each column individually, with a
    window size. The rolling function is given by rolling_type, for example 'mean',
    'median', 'sum', etcetera. The window size can be given directly by window_size, or
    indirectly by values_roll, which is then converted by unit_roll and freq. values_roll
    and window_size can be array-like, in which case each window size will be used alongside
    each other.

    An important note is that this transformer assumes that the data is sorted by time already!
    So if the input dataframe is not sorted by time (in ascending order), the results will be
    completely wrong.

    A note about the way the output is rolled: in case of 'lag' and 'diff', the output will
    always be lagged, even if lookback is 0. This is because these functions inherently look
    at a previous cell, regardless of what the lookback is. All other functions will start
    by looking at the current cell if lookback is 0. (and will also look at previous cells
    if window_size is greater than 1)

    'ewm' looks at window_size a bit different: instead of a discrete number of points to
    look at, 'ewm' needs a parameter alpha between 0 and 1 instead.

    Parameters
    ----------
    rolling_type : string, optional (default="mean")
        The rolling function. Must be one of: 'median', 'skew', 'kurt', 'max', 'std', 'lag',
        'mean', 'diff', 'sum', 'var', 'min', 'numpos', 'ewm', 'fourier', 'cwt'
    lookback : number type, optional (default=1)
        the features that are built will be shifted by this value
        If more than 0, this prevents leakage
    values_roll : array-like, shape = (n_outputs, ), optional (default=None)
        vector of lag or rolling values in specified unit
    unit_roll : string, optional (default=None)
        unit of values_roll, must be parseable by utils.time.unit_to_seconds
    freq : string, optional (default=None)
        freq of measurements in values_roll, shoud match ^(\d+) (\w+)$
    window_size : array-like, shape = (n_outputs, ), optional (defaut=None)
        vector of values to shift. Ignored when rolling_type is ewm
    deviation : str, optional (default=None)
        one of ['subtract', 'divide']. If this option is set, the resulting column will either
        have the original column subtracted, or will be divided by the original column. If None,
        just return the resulting column. This option is not allowed when rolling_type is 'cwt'
        or 'fourier', but it is allowed with all other rolling_types.
    alpha : numeric, optional (default=0.5)
        if rolling_type is 'ewm', this is the parameter alpha used for weighing the samples.
        The current sample weighs alpha, the previous sample weighs alpha*(1-alpha), the
        sample before that weighs alpha*(1-alpha)^2, etcetera. Must be in (0, 1]
    width : numeric, optional (default=1)
        if rolling_type is 'cwt', the wavelet transform uses a ricker signal. This parameter
        defines the width of that signal
    keep_original : boolean, optional (default=True)
        if the original columns should be kept or discarded
        True by default, which means the new columns are added to the old ones

    Examples
    --------
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> import pandas as pd
    >>> df = pd.DataFrame({'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    >>>                    'DEBIET': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3]})
    >>>
    >>> BuildRollingFeatures(rolling_type='lag', window_size = [0,1,4], \\
    >>>                      lookback=0, keep_original=False).fit_transform(df)
                RAIN#lag_0  DEBIET#lag_0    RAIN#lag_1  DEBIET#lag_1    RAIN#lag_4  DEBIET#lag_4
    0           0.1         1               NaN         NaN             NaN         NaN
    1           0.2         2               0.1         1.0             NaN         NaN
    2           0.0         3               0.2         2.0             NaN         NaN
    3           0.6         4               0.0         3.0             NaN         NaN
    4           0.1         5               0.6         4.0             0.1         1.0
    5           0.0         5               0.1         5.0             0.2         2.0
    6           0.0         4               0.0         5.0             0.0         3.0
    7           0.0         3               0.0         4.0             0.6         4.0
    8           0.0         2               0.0         3.0             0.1         5.0
    9           0.0         4               0.0         2.0             0.0         5.0
    10          0.0         2               0.0         4.0             0.0         4.0
    11          0.0         3               0.0         2.0             0.0         3.0

    """

    def _validate_params(self):
        """apply various checks to the inputs of the __init__ function
        throw value error or type error based on the result
        """

        if self.window_size is None and \
                (self.values_roll is None or
                 self.unit_roll is None or self.freq is None) and \
                self.rolling_type != "ewm":
            raise ValueError(("Either window_size must not be None, or values_roll,"
                              "unit_roll, freq must all be not None"))

        # we allow scalar inputs, but _calc_window_size expects an interable
        if isinstance(self.values_roll, (int, float)):
            self._values_roll = [self.values_roll]
        else:
            self._values_roll = self.values_roll

        if not (isinstance(self.unit_roll, str) or self.unit_roll is None):
            raise TypeError("unit_roll must be a string")
        if not (isinstance(self.freq, str) or self.freq is None):
            raise TypeError("freq must be a string")
        if not isinstance(self.lookback, (int, float)):
            raise TypeError("lookback must be a scalar")
        if self.lookback < 0:
            raise ValueError("lookback cannot be negative!")
        if not (isinstance(self._values_roll, (list, tuple)) or
                (type(self._values_roll) is np.ndarray and self._values_roll.ndim == 1) or
                self._values_roll is None):
            raise TypeError("values_roll must be a scalar, list or 1d numpy array")
        if not isinstance(self.rolling_type, str):
            raise TypeError("rolling_type must be a string")
        if not isinstance(self.keep_original, bool):
            raise TypeError("keep_original must be a boolean")
        if not isinstance(self.width, (int, float)):
            raise TypeError("width must be a scalar")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a scalar")
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")

        if self.freq is not None:
            regex_result = re.search(r"^(\d+) ?(\w+)$", self.freq)
            if regex_result is None:
                raise ValueError("The frequency '%s' must be of the form 'num unit', \
                                  where num is an integer and unit is a string" % self.freq)

        if not (isinstance(self.window_size, (list, tuple)) or
                (type(self.window_size) is np.ndarray and self.window_size.ndim == 1) or
                self.window_size is None or
                isinstance(self.window_size, (int, float))):
            raise TypeError("window_size must be None, numeric, list or 1d numpy array")

        if self.deviation is not None and self.rolling_type in ["fourier", "cwt"]:
            raise ValueError("Deviation cannot be used together with {}".format(self.rolling_type))
        if self.deviation not in [None, "subtract", "divide"]:
            raise ValueError("Deviation must be one of [None, 'subtract', 'divide']")

    def _calc_window_size(self, values_roll, unit_roll, freq):
        """Determine number of rows to use in rolling
        Based on a frequency of a dataframe and desired rolling horizon,
        determing how many rows this is, taking into account the correct units

        Parameters
        ----------
        values_roll : array-like, shape = (n_outputs, )
            vector of lag or rolling values in specified unit
        unit_roll : string
            unit of values_roll, must be parseable by utils.time.unit_to_seconds
        freq : string
            freq of measurements in values_roll, should match ^(\d+) (\w+)$
            where the word at the end must be parseable by utils.time.unit_to_seconds

        Returns
        -------
        window_sizes : array-like, shape = (n_outputs, )
            vector of values to use as window sizes
        """

        regex_result = re.search(r"^(\d+) ?(\w+)$", freq)
        freq_timestep, freq_unit = regex_result.groups()

        # The difference in units between unit_roll and freq.
        # for example, if unit_roll is "day" and freq = "30 min", then diff_unit will be 48
        diff_unit = (unit_to_seconds(unit_roll) / unit_to_seconds(freq_unit)) / \
            float(freq_timestep)
        shifts = [val * diff_unit for val in values_roll]

        if not all([shift.is_integer() for shift in shifts]):
            raise ValueError(("The frequency '%s', does not evenly divide"
                              " into all the rollling units ('%s')")
                             % (freq, unit_roll))
        return([int(shift) for shift in shifts])

    def _get_rolling_fun(self, rolling_type="mean"):
        """Given a function name as a string, creates a function that
        applies that rolling function

        Parameters
        ----------
        rolling_type : string, default="mean"
            the description of the rolling function. must be one of lag, sum, mean,
            median, var, std, max, min, skew, kurt, diff, numpos, ewm, fourier, cwt

        Returns
        -------
        rolling_function : function
            function with two inputs: a pandas series and an integer. Will apply
            some rolling function to the series, with window size of the integer.
            Alternatively, in case of fourier/cwt, a function with one input:
            a numpy array. Will output another numpy array.
        """
        # https://pandas.pydata.org/pandas-docs/stable/api.html#window
        rolling_functions = {
            "lag": lambda arr, n: arr.shift(n),
            "sum": lambda arr, n: arr.rolling(n).sum(),
            "mean": lambda arr, n: arr.rolling(n).mean(),
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
            "cwt": lambda vector:  signal.cwt(vector, signal.ricker, [self.width])[0],
        }

        if rolling_type not in rolling_functions:
            raise ValueError("The rolling_type is %s, which is not an available function"
                             % rolling_type)
        return(rolling_functions[rolling_type])

    def __init__(self, rolling_type="mean", lookback=1, values_roll=None, unit_roll=None,
                 freq=None, window_size=None, deviation=None, alpha=0.5, width=1,
                 keep_original=True):
        self.values_roll = values_roll
        self.unit_roll = unit_roll
        self.freq = freq
        self.window_size = window_size
        self.lookback = lookback
        self.rolling_type = rolling_type
        self.deviation = deviation
        self.alpha = alpha
        self.width = width
        self.keep_original = keep_original
        logger.debug("Initialized rolling generator. rolling_type={}, lookback={}, "
                     "values_roll={}, unit_roll={}, freq={}, window_size={}, "
                     "deviation={}, alpha={}, width={}, keep_original={}".
                     format(rolling_type, lookback, values_roll, unit_roll, freq, window_size,
                            deviation, alpha, width, keep_original))

    def fit(self, X=None, y=None):
        """Calculates window_size and feature function

        Parameters
        ----------
        X : optional, is ignored
        y : optional, is ignored
        """

        self._validate_params()

        if self.rolling_type == "ewm":
            # ewm needs no window_size calculation
            self.window_size_ = "ewm"
            self.suffix_ = ["ewm_" + str(self.alpha)]
        elif self.window_size is None:
            self.window_size_ = self._calc_window_size(
                                    self._values_roll, self.unit_roll, self.freq)
            self.suffix_ = [self.rolling_type + "_" + str(i) + "_" + self.unit_roll
                            for i in self._values_roll]
        else:
            self.window_size_ = self.window_size
            # Singleton window_size is also allowed
            if isinstance(self.window_size_, (int, float)):
                self.window_size_ = [self.window_size_]
            self.suffix_ = [self.rolling_type + "_" + str(window_size)
                            for window_size in self.window_size_]

        self.rolling_fun_ = self._get_rolling_fun(self.rolling_type)
        logger.debug("Done fitting transformer. window size: {}, suffix: {}".
                     format(self.window_size_, self.suffix_))
        return self

    def _apply_deviation(self, arr, original, deviation):
        """Helper function to apply deviation during the transform"""
        if deviation is None:
            return arr
        if deviation == "subtract":
            return arr - original
        if deviation == "divide":
            # Pandas will insert inf when dividing by 0
            return arr / original

    def transform(self, X):
        """Transforms pandas dataframe X to apply rolling function

        Parameters
        ----------
        X : pandas dataframe, shape = (n_rows, n_features)
           the pandas dataframe that you want to apply rolling functions on

        Returns
        -------
        result : pandas dataframe, shape = (n_rows, n_features * (n_outputs + 1))
            the pandas dataframe, appended with the new columns
        """
        # X = check_array(X) # dont do this, we want to use pandas only
        check_is_fitted(self, 'window_size_')
        if self.keep_original:
            result = X.copy()
        else:
            result = pd.DataFrame(index=X.index)

        if self.rolling_type in ["fourier", "cwt"]:
            for window_size, suffix in zip(self.window_size_, self.suffix_):

                for column in X.columns:
                    new_features = multicol_output(X[column], window_size, self.rolling_fun_,
                                                   self.rolling_type == "fourier") \
                                                   .shift(self.lookback)
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
                new_features = X.apply(lambda arr: self._apply_deviation(
                    self.rolling_fun_(arr, window_size).shift(self.lookback),
                    arr,
                    self.deviation
                ))
                new_features.columns = ["#".join([str(col), suffix])
                                        for col in new_features.columns]
                result = pd.concat([result, new_features], axis=1)

        self._feature_names = list(result.columns.values)
        log_new_columns(result, X)
        log_dataframe_characteristics(result, logging.DEBUG)  # Log types as well
        return(result)

    def get_feature_names(self):
        """
        Returns feature names for the outcome of the last transform call.
        """
        check_is_fitted(self, '_feature_names')
        return self._feature_names
