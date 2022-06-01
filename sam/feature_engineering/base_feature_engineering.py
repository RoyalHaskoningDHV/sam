import logging

from sklearn.base import TransformerMixin


class BaseFeatureEngineer(TransformerMixin):
    """
    Base class for feature engineering.
    To use this class, you need to implement the feature_engineer method.

    """

    def __init__(
        self,
    ):
        pass

    def feature_engineer(self, X):
        raise NotImplementedError

    def fit(
        self,
        X,
        y=None,
    ):
        return self

    def transform(self, X):
        logging.info("Feature engineering - input shape: %s", X.shape)
        X_out = self.feature_engineer(X)
        logging.info("Feature engineering - output shape: %s", X_out.shape)
        return X_out


class FeatureEngineerFromFunction(BaseFeatureEngineer):
    """
    Feature engineering from a function.

    Parameters
    ----------
    feature_engineer : function
        Function to apply to the data.
    """

    def __init__(
        self,
        feature_engineer: callable,
    ):
        super().__init__()
        self.feature_engineer = feature_engineer
