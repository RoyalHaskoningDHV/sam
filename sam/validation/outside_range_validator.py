from typing import Union
import numpy as np
import pandas as pd

from sam.validation import BaseValidator


class OutsideRangeValidator(BaseValidator):
    """
    Validator class method that removes data that is outside the provided range

    Parameters
    ----------
    cols: list (optional)
        Columns of input data to be checkout for being outside range. If None, all columns will be
        validated
    min_value: float, dict or "auto" (optional)
        Minimum value to check against. If None, no minimum will be checked. If "auto", the minimum
        value of the data will be used.
    max_value: float, dict or "auto" (optional)
        Maximum value to check against. If None, no maximum will be checked. If "auto", the maximum
        value of the data will be used.
    """

    def __init__(
        self,
        cols: list = None,
        min_value: Union[float, dict, str] = None,
        max_value: Union[float, dict, str] = None,
    ):
        self.cols = cols
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe containing the features to be checked.
        y: pd.Series or pd.DataFrame (optional)
            Series or dataframe containing the target (ignored)
        """

        if self.cols is None:
            self.cols = X.columns

        if self.min_value == "auto":
            self.min_value_ = X[self.cols].min().to_dict()
        elif self.min_value is None:
            self.min_value_ = -np.inf
        elif isinstance(self.min_value, str):
            raise ValueError("min_value must be a float, dict or 'auto'")
        else:
            self.min_value_ = self.min_value

        if self.max_value == "auto":
            self.max_value_ = X[self.cols].max().to_dict()
        elif self.max_value is None:
            self.max_value_ = np.inf
        elif isinstance(self.max_value, str):
            raise ValueError("max_value must be a float, dict or 'auto'")
        else:
            self.max_value_ = self.max_value

        self._feature_names_out = X.columns

        return self

    def validate(self, X):
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe containing the features to be checked.
        """

        invalid_data = pd.DataFrame(
            data=np.zeros_like(X.values).astype(bool),
            index=X.index,
            columns=X.columns,
        )

        invalid_data[self.cols] = X[self.cols].gt(self.max_value_) | X[self.cols].lt(
            self.min_value_
        )

        return invalid_data
