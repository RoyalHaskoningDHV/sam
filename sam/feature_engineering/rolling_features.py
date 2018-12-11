import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sam.utils.time import unit_to_seconds


def fourier(arr, n):
    class FFTHelper:
        # https://stackoverflow.com/a/39064656
        def __init__(self, nrow, ncol):
            self.series = np.empty((nrow, ncol))
            self.series[:] = np.NaN
            self.calls = ncol - 1

        def calc_fft(self, vector):
            values = np.absolute(np.fft.fft(vector))
            self.series[self.calls, :] = values
            self.calls = self.calls + 1
            return np.NaN  # return something to make Rolling apply not error
    helper = FFTHelper(len(arr), n)
    arr.rolling(n).apply(helper.calc_fft, raw=True)
    return pd.DataFrame(helper.series)


class BuildRollingFeatures(BaseEstimator, TransformerMixin):
    """Applies some rolling function to a pandas dataframe

    This class provides a stateless transformer that applies to each column in a dataframe.
    It works by applying a certain rolling function to each column individually, with a window size.
    The rolling function is given by rolling_type, for example 'mean', 'median', 'sum', etcetera
    The window size can be given directly by shift, or indirectly by values_roll, which is then
    converted by unit_roll and freq. values_roll and shift can be array-like, in which case each
    window size will be used alongside each other.

    An important note is that this transformer assumes that the data is sorted by time already!
    So if the input dataframe is not sorted by time (in ascending order), the results will be
    completely wrong.

    Parameters
    ----------
    rolling_type : string, optional (default="mean")
    lookback : number type, optional (default=1)
        the features that are built will be shifted by this value. If more than 0, this prevents leakage
    values_roll : array-like, shape = (n_outputs, ), optional (default=None)
        vector of lag or rolling values in specified unit
    unit_roll : string, optional (default=None)
        unit of values_roll, must be parseable by utils.time.unit_to_seconds
    freq : string, optional (default=None)
        freq of measurements in values_roll, shoud match ^(\d+) (\w+)$
    shift : array-like, shape = (n_outputs, ), optional (defaut=None)
        vector of values to shift
    keep_original : boolean, optional (default=True)
        if the original columns should be kept or discarded
        True by default, which means the new columns are added to the old ones
    """

    def _validate_params(self):
        """apply various checks to the inputs of the __init__ function
        throw value error or type error based on the result
        """

        if self.shift is None and (self.values_roll is None or self.unit_roll is None or self.freq is None):
            raise ValueError("Either shift must not be None, or values_roll, unit_roll, freq must all be not None")

        # we allow scalar inputs, but _calc_shift expects an interable
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
        if not (isinstance(self._values_roll, (list, tuple)) or
                (type(self._values_roll) is np.ndarray and self._values_roll.ndim == 1) or
                self._values_roll is None):
            raise TypeError("values_roll must be a scalar, list or 1d numpy array")
        if not isinstance(self.rolling_type, str):
            raise TypeError("rolling_type must be a string")
        if not isinstance(self.keep_original, bool):
            raise TypeError("keep_original must be a boolean")

        if self.freq is not None:
            regex_result = re.search("^(\d+) ?(\w+)$", self.freq)
            if regex_result is None:
                raise ValueError("The frequency '%s' must be of the form 'num unit', \
                                  where num is an integer and unit is a string" % self.freq)

        if not (isinstance(self.shift, (list, tuple)) or
                (type(self.shift) is np.ndarray and self.shift.ndim == 1) or
                self.shift is None or
                isinstance(self.shift, (int, float))):
            raise TypeError("shift must be None, numeric, list or 1d numpy array")

    def _calc_shift(self, values_roll, unit_roll, freq):
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
        shifts : array-like, shape = (n_outputs, )
            vector of values to shift
        """

        regex_result = re.search("^(\d+) ?(\w+)$", freq)
        freq_timestep, freq_unit = regex_result.groups()

        # The difference in units between unit_roll and freq.
        # for example, if unit_roll is "day" and freq = "30 min", then diff_unit will be 48
        diff_unit = (unit_to_seconds(unit_roll) / unit_to_seconds(freq_unit)) / float(freq_timestep)
        shift = [val * diff_unit for val in values_roll]

        if not all([foo.is_integer() for foo in shift]):
            raise ValueError("The frequency '%s', does not evenly divide into all the rollling units ('%s')"
                             % (freq, unit_roll))
        return([int(foo) for foo in shift])

    def _get_rolling_fun(self, rolling_type="mean"):
        """Given a function name as a string, creates a function that
        applies that rolling function

        Parameters
        ----------
        rolling_type : string, default="mean"
            the description of the rolling function. must be one of lag, sum, mean,
            median, var, std, max, min, skew, kurt, diff, numpos

        Returns
        -------
        rolling_function : function
            function with two inputs: a pandas series and an integer. Will apply
            some rolling function to the series, with window size of the integer
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
            "fourier": fourier
        }

        if rolling_type not in rolling_functions:
            raise ValueError("The rolling_type is %s, which is not an available function" % rolling_type)
        return(rolling_functions[rolling_type])

    def __init__(self, rolling_type="mean", lookback=1, values_roll=None, unit_roll=None,
                 freq=None, shift=None, keep_original=True):
        self.values_roll = values_roll
        self.unit_roll = unit_roll
        self.freq = freq
        self.shift = shift
        self.lookback = lookback
        self.rolling_type = rolling_type
        self.keep_original = keep_original

    def fit(self, X=None, y=None):
        """Calculates shift and feature function

        Parameters
        ----------
        X : optional, is ignored
        y : optional, is ignored
        """

        self._validate_params()

        if self.shift is None:
            self.shift_ = self._calc_shift(self._values_roll, self.unit_roll, self.freq)
            self.suffix_ = [self.rolling_type + "_" + str(i) + "_" + self.unit_roll for i in self._values_roll]
        else:
            self.shift_ = self.shift
            # Singleton shift is also allowed
            if isinstance(self.shift_, (int, float)):
                self.shift_ = [self.shift_]
            self.suffix_ = [self.rolling_type + "_" + str(shift) for shift in self.shift_]

        self.rolling_fun_ = self._get_rolling_fun(self.rolling_type)
        return self

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
        check_is_fitted(self, 'shift_')
        if self.keep_original:
            result = X.copy()
        else:
            result = pd.DataFrame(index=X.index)

        if self.rolling_type is "fourier":
            for shift, suffix in zip(self.shift_, self.suffix_):
                for column in X.columns:
                    foo = fourier(X[column], shift).shift(self.lookback)
                    foo.columns = ["_".join([str(column), suffix, str(j)]) for j in range(shift)]
                    foo = foo.set_index(X.index)
                    result = pd.concat([result, foo], axis=1)
        else:
            for shift, suffix in zip(self.shift_, self.suffix_):
                foo = X.apply(lambda arr: self.rolling_fun_(arr, shift).shift(self.lookback))
                foo.columns = ["_".join([str(col), suffix]) for col in foo.columns]
                result = pd.concat([result, foo], axis=1)
        
        self._feature_names = list(result.columns.values)

        return(result)

    def get_feature_names(self):
        """
        Returns feature names for last transform call
        """
        check_is_fitted(self, '_feature_names')
        return self._feature_names
