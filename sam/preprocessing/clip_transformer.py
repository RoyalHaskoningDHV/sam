from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClipTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that clips values to a given range.

    Parameters
    ----------
    cols: list (optional)
        Columns of input data to be clipped. If None, all columns will be clipped.
    min_value: float (optional)
        Minimum value to clip to. If None, min will be set to the minimum value of the data.
    max_value: float (optional)
        Maximum value to clip to. If None, max will be set to the maximum value of the data.
    """

    def __init__(
        self,
        cols: list = None,
        min_value: float = None,
        max_value: float = None,
    ):
        self.cols = cols
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series] = None):
        """
        Fit the transformer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe containing the features to be clipped.
        y: pd.Series or pd.DataFrame (optional)
            Series or dataframe containing the target (ignored)
        """

        if self.cols is None:
            self.cols = X.columns

        if self.min_value is None:
            self.min_value = X[self.cols].min().to_dict()

        if self.max_value is None:
            self.max_value = X[self.cols].max().to_dict()

        self._feature_names_out = X.columns

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe containing the features to be clipped.
        """

        X = X.copy()
        X[self.cols] = X[self.cols].clip(self.min_value, self.max_value)
        return X

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get the names of the output features.
        """

        return self._feature_names_out
