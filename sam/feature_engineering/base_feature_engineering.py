import logging

from sklearn.base import TransformerMixin


class BaseFeatureEngineer(TransformerMixin):
    """
    Base class for feature engineering.
    To use this class, you need to implement the feature_engineer method.

    """

    def __init__(self):
        pass

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

    def feature_engineer_(self, X, y=None):
        raise NotImplementedError("You need to implement the feature_engineer method.")

    def fit(self, X, y=None):
        X_out = self.transform(X)
        self._feature_names = X_out.columns.tolist()
        return self

    def transform(self, X, y=None):
        logging.info("Feature engineering - input shape: %s", X.shape)
        X_out = self.feature_engineer_(X, y)
        logging.info("Feature engineering - output shape: %s", X_out.shape)
        return X_out
