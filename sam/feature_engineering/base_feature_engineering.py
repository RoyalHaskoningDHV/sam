from ast import Call
import logging

import pandas as pd
from sklearn.base import TransformerMixin
from abc import ABC, abstractmethod


class BaseFeatureEngineer(TransformerMixin, ABC):
    """
    Base class for feature engineering.
    To use this class, you need to implement the feature_engineer method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def feature_engineer_(self, X, y=None) -> pd.DataFrame:
        raise NotImplementedError("You need to implement the feature_engineer method.")

    def fit(self, X, y=None):
        X_out = self.transform(X)
        self._feature_names = X_out.columns.tolist()
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        logging.info("Feature engineering - input shape: %s", X.shape)
        X_out = self.feature_engineer_(X, y)
        logging.info("Feature engineering - output shape: %s", X_out.shape)
        return X_out


class FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering class. This class is used to feature engineer the data using default
    methods. You can implement your own feature engineering code and use the `_from_function`
    method to create a new feature engineering transformer.

    Example
    -------
    >>> from sam.feature_engineering.base_feature_engineering import FeatureEngineer
    >>> def feature_engineer(X, y=None):
    >>>     X['C'] = X['A'] + X['B']
    >>>     return X
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]})
    >>> fe = FeatureEngineer._from_function(feature_engineering)
    >>> df_out = fe.fit_transform(df)
    """

    @classmethod
    def _from_function(cls, feature_engineer: callable):
        """
        Create a BaseFeatureEngineer from a function.

        This function could be any feature engineering function that takes a pandas DataFrame X
        and target y as input and returns a pandas DataFrame as output.

        """
        self = cls()
        self.feature_engineer_ = feature_engineer
        return self
