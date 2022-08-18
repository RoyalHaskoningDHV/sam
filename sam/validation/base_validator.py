import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class BaseValidator(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for validators"""

    def __init__(self):
        pass

    @abstractmethod
    def validate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the data.

        This method should return a boolean array of the same shape as X, where True indicates a
        value that is invalid.
        """
        raise NotImplementedError("You need to implement the validate method.")

    def fit(self, X, y=None):
        """fit method"""
        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = self.feature_names_in_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """transform method"""
        X = X.copy()
        invalid_data = self.validate(X)
        X[invalid_data] = np.nan
        return X

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Function for obtaining feature names. Generally used instead of the attribute, and more
        compatible with the sklearn API.
        Returns
        -------
        list:
            list of feature names
        """
        check_is_fitted(self, "_feature_names_out_")
        return self._feature_names_out_
