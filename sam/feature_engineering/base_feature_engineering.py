import logging
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from sklearn.base import TransformerMixin


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
        self._feature_names = self.feature_engineer_(X, y).columns.tolist()
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        logging.info("Feature engineering - input shape: %s", X.shape)
        X_out = self.feature_engineer_(X, y)
        logging.info("Feature engineering - output shape: %s", X_out.shape)
        return X_out

    def get_feature_names(self) -> list:
        """
        Function for obtaining feature names. Generally used instead of the attribute, and more
        compatible with the sklearn API.

        Returns
        -------
        list:
            list of feature names
        """
        # check_is_fitted(self.feature_engineer_, "_feature_names")
        return self._feature_names


class FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineering class. This class is used to feature engineer the data using default
    methods. You can implement your own feature engineering code and use the `_from_function`
    method to create a new feature engineering transformer.

    Parameters
    ----------
    feature_engineer_function : Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
        The feature engineering function.

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

    def __init__(
        self,
        feature_engineer_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = None,
    ):
        self.feature_engineer_function = feature_engineer_function

    def feature_engineer_(self, X, y=None) -> pd.DataFrame:
        if self.feature_engineer_function is None:
            raise ValueError("You need to specify a feature engineering function.")
        return self.feature_engineer_function(X, y)


class IdentityFeatureEngineer(BaseFeatureEngineer):
    """
    Identity feature engineering class. This is a placeholder class for when you don't want to
    apply any feature engineering. Makes compatibility with the sam API easier.

    Parameters
    ----------
    feature_engineer_function : Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
        The feature engineering function.

    Example
    -------
    >>> from sam.feature_engineering.base_feature_engineering import IdentityFeatureEngineer
    >>> df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]})
    >>> fe = IdentityFeatureEngineer()
    >>> df_out = fe.fit_transform(df)
    """

    def __init__(self):
        pass

    def feature_engineer_(self, X, y=None) -> pd.DataFrame:
        return X
