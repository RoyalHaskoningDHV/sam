import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseFeatureEngineer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for feature engineering.
    To use this class, you need to implement the feature_engineer method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def feature_engineer_(self, X) -> pd.DataFrame:
        """Implement this method to do the feature engineering."""
        raise NotImplementedError("You need to implement the feature_engineer_ method.")

    def fit(self, X, y=None):
        """fit method"""
        self._feature_names = self.feature_engineer_(X).columns.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        """transform method"""
        logging.info("Feature engineering - input shape: %s", X.shape)
        X_out = self.feature_engineer_(X)
        logging.info("Feature engineering - output shape: %s", X_out.shape)
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Function for obtaining feature names. Generally used instead of the attribute, and more
        compatible with the sklearn API.

        Returns
        -------
        list:
            list of feature names
        """
        check_is_fitted(self, "_feature_names")
        return self._feature_names


class FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering class. This class is used to feature engineer the data using default
    methods and makes integration with the timeseries models easier. You can implement your own
    feature engineering code as a function that takes two arguments: X and y and returns a
    feature table as a pandas dataframe.

    Parameters
    ----------
    feature_engineer_function : Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
        The feature engineering function.

    Example
    -------
    >>> from sam.feature_engineering.base_feature_engineering import FeatureEngineer
    >>> def feature_engineer(X, y=None):
    ...     X['C'] = X['A'] + X['B']
    ...     return X
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]})
    >>> fe = FeatureEngineer(feature_engineer)
    >>> df_out = fe.fit_transform(df)
    """

    def __init__(
        self,
        feature_engineer_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = None,
    ):
        self.feature_engineer_function = feature_engineer_function

    def feature_engineer_(self, X: pd.DataFrame) -> pd.DataFrame:
        """feature engineering function"""
        if self.feature_engineer_function is None:
            raise ValueError("You need to specify a feature engineering function.")
        return self.feature_engineer_function(X)


class IdentityFeatureEngineer(BaseFeatureEngineer):
    """
    Identity feature engineering class. This is a placeholder class for when you don't want to
    apply any feature engineering. Makes compatibility with the sam API easier.

    Parameters
    ----------
    numeric_only : bool
        Whether to only include numeric columns in the output.

    Example
    -------
    >>> from sam.feature_engineering.base_feature_engineering import IdentityFeatureEngineer
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]})
    >>> fe = IdentityFeatureEngineer()
    >>> df_out = fe.fit_transform(df)
    """

    def __init__(self, numeric_only: bool = True):
        self.numeric_only = numeric_only

    def feature_engineer_(self, X: pd.DataFrame) -> pd.DataFrame:
        """feature engineering function, returns the input dataframe"""
        if self.numeric_only:
            return X.select_dtypes(include=np.number)
        return X
