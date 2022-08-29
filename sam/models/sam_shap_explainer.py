from typing import Callable

import numpy as np
import pandas as pd

from sam.models.base_model import BaseTimeseriesRegressor


class SamShapExplainer(object):
    """
    An object that imitates a SHAP explainer object. (Sort of) implements the base Explainer
    interface which can be found here
    <https://github.com/slundberg/shap/blob/master/shap/explainers/explainer.py>.
    The more advanced, tensorflow-specific attributes can be accessed with obj.explainer.
    The reason the interface is only sort of implemented, is the same reason why
    MLPTimeseriesRegressor doesn't entirely implement the skearn interface - for predicting, y is
    needed, which is not supported by the SamShapExplainer.

    Parameters
    ----------
    explainer: shap TFDeepExplainer object
        A shap explainer object. This will be used to generate the actual shap values
    model: BaseTimeseriesRegressor model
        This will be used to do the preprocessing before calling explainer.shap_values
    """

    def __init__(
        self, explainer: Callable, model: BaseTimeseriesRegressor, preprocess_predict: Callable
    ) -> None:
        self.explainer = explainer
        self.preprocess_predict = preprocess_predict

        # Create a proxy model that can call only 3 attributes we need
        class SamProxyModel:
            fit = None
            feature_names_ = model.get_feature_names_out()
            preprocess_predict = model.preprocess_predict

        self.model = SamProxyModel()
        # Trick sklearn into thinking this is a fitted variable
        self.model.feature_engineer_ = model.feature_engineer_
        # Will likely be somewhere around 0
        self.expected_value = explainer.expected_value

    def shap_values(self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs) -> np.ndarray:
        """
        Imitates explainer.shap_values, but combined with the preprocessing from the model.
        Returns a similar format as a regular shap explainer: a list of numpy arrays, one
        for each output of the model.

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe used to 'train' the explainer
        y: pd.Series, optional (default=None)
            Target data used to 'train' the explainer.
        """
        X_transformed = self.model.preprocess_predict(X, y, dropna=False)
        return self.explainer.shap_values(X_transformed, *args, **kwargs)

    def attributions(self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs) -> np.ndarray:
        """
        Imitates explainer.attributions, which by default just mirrors shap_values

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe used to 'train' the explainer
        y: pd.Series, optional (default=None)
            Target data used to 'train' the explainer.
        """
        return self.shap_values(X, y, *args, **kwargs)

    def test_values(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Only the preprocessing from the model, without the shap values.
        Returns a pandas dataframe with the actual values used for the explaining
        Can be used to better interpret the numpy array that is outputted by shap_values.
        For example, if shap_values outputs a 5x20 numpy array, that means you explained 5 objects
        with 20 features. This function will then return a 5x20 pandas dataframe.

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe used to 'train' the explainer
        y: pd.Series, optional (default=None)
            Target data used to 'train' the explainer.
        """
        X_transformed = self.model.preprocess_predict(X, y, dropna=False)
        return pd.DataFrame(X_transformed, columns=self.model.feature_names_, index=X.index)
