import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as smapi
from sam.metrics import tilted_loss


class LinearQuantileRegression(BaseEstimator, RegressorMixin):
    """ scikit-learn style wrapper for QuantReg
    Fits a linear quantile regression model, copied from
    https://github.com/Marco-Santoni/skquantreg/blob/master/skquantreg/quantreg.py

    Parameters
    ----------
        quantiles: list or float, default=[0.05, 0.95]
            Quantiles to fit, with `` 0 < q < 1 `` for each q in quantiles.
        tol: float, default=1e-3
            The tolerance for the optimization. The optimization stops
            when duality gap is smaller than the tolerance
        max_iter: int, default=1000
            The maximum number of iterations

    Attributes
    ----------
    model_: statsmodel model
        The underlying statsmodel class
    model_result_: statsmodel results
        The underlying statsmodel results

    Examples
    --------
    >>> from sam.models import LinearQuantileRegression
    >>> from sam.data_sources import read_knmi
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', freq='hourly',
    >>>                 variables=['FH', 'FF', 'FX', 'T'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> # Fit model
    >>> model = LinearQuantileRegression()
    >>> model.fit(X, y)
    """
    def __init__(self, quantiles=[0.05, 0.95], tol=1e-3, max_iter=1000):
        self.quantiles = quantiles
        self.tol = tol
        self.max_iter = max_iter

    def _fit_single_model(self, X, y, q):
        # Statsmodels requires to add constant columns manually
        # otherwise the models will not fit an intercept
        model_ = QuantReg(y, smapi.add_constant(X))
        model_result_ = model_.fit(q, p_tol=self.tol, max_iter=self.max_iter)
        model_result_
        return model_result_

    def fit(self, X, y):
        """ Fit a Linear Quantile Regression using statsmodels
        """
        if type(self.quantiles) is float:
            self.q_ = [self.quantiles]
        elif type(self.quantiles) is list:
            self.q_ = self.quantiles
        else:
            raise TypeError(f'Invalid type, quantile {self.quantiles} '
                            f'should be a float or list of floats')
        self.prediction_cols = [f'predict_q_{q}' for q in self.quantiles]
        self.models_ = [self._fit_single_model(X, y, q) for q in self.quantiles]
        return self

    def predict(self, X):
        """ Predict / estimate quantiles
        """
        preds = [m.predict(smapi.add_constant(X)) for m in self.models_]
        preds_df = pd.concat(preds, axis=1)
        preds_df.columns = self.prediction_cols
        return preds_df

    def score(self, X, y):
        """ Default score Function. Returns the tilted loss
        """
        y_pred = self.predict(X)
        scores = [tilted_loss(
            y_true=y,
            y_pred=y_pred[f'predict_q_{q}'],
            quantile=q) for q in self.quantiles]
        score = np.mean(scores)
        return score
