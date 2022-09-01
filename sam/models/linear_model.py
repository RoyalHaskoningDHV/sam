import numpy as np
import pandas as pd
from sam.metrics import tilted_loss
from sklearn.base import BaseEstimator, RegressorMixin
from sam.utils.warnings import add_future_warning

# Keep package independent of statsmodels
try:
    from statsmodels.regression.quantile_regression import QuantReg
except ImportError:
    pass


class LinearQuantileRegression(BaseEstimator, RegressorMixin):
    """
    scikit-learn style wrapper for QuantReg
    Fits a linear quantile regression model, base idea from
    https://github.com/Marco-Santoni/skquantreg/blob/master/skquantreg/quantreg.py
    This class requires statsmodels

    Parameters
    ----------
    quantiles: list or float, default=[0.05, 0.95]
        Quantiles to fit, with `` 0 < q < 1 `` for each q in quantiles.
    tol: float, default=1e-3
        The tolerance for the optimization. The optimization stops
        when duality gap is smaller than the tolerance
    max_iter: int, default=1000
        The maximum number of iterations
    fit_intercept: bool, default=True
        Whether to calculate the intercept for this model. If set to false,
        no intercept will be used in calculations (e.g. data is expected to be
        already centered).

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
    ...                 variables=['FH', 'FF', 'FX', 'T']).set_index('TIME')
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> # Fit model
    >>> model = LinearQuantileRegression()
    >>> model.fit(X, y)  # doctest: +SKIP
    """

    @add_future_warning(
        "This class is deprecated and will be removed in a future version. "
        "Please use the LassoTimeseriesRegressor class from the sam.models "
        "or the QuantileRegressor class from sklearn.linear_model instead."
    )
    def __init__(
        self,
        quantiles: list = [0.05, 0.95],
        tol: float = 1e-3,
        max_iter: int = 1000,
        fit_intercept: bool = True,
    ):
        self.quantiles = quantiles
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def _fit_single_model(self, X, y, q):
        # Statsmodels requires to add constant columns manually
        # otherwise the models will not fit an intercept
        X = X.copy()
        if self.fit_intercept:
            X = X.assign(const=1)  # f(x) = a + bX = a*const + b*X
        model_ = QuantReg(y, X)
        model_result_ = model_.fit(q, p_tol=self.tol, max_iter=self.max_iter)
        coef = model_result_.params
        pvalues = model_result_.pvalues
        return coef, pvalues

    def _array_to_df(self, X):
        """Transform to dataframe"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"X{i}" for i in range(1, X.shape[1] + 1)]
        return X

    def fit(self, X: np.array, y: np.array):
        """
        Fit a Linear Quantile Regression using statsmodels
        """
        if type(self.quantiles) is float:
            self.q_ = [self.quantiles]
        elif type(self.quantiles) is list:
            self.q_ = self.quantiles
        else:
            raise TypeError(
                f"Invalid type, quantile {self.quantiles} " f"should be a float or list of floats"
            )
        self.prediction_cols = [f"predict_q_{q}" for q in self.quantiles]

        X = self._array_to_df(X)
        self.feature_names_in_ = X.columns.tolist()
        models_ = [self._fit_single_model(X, y, q) for q in self.quantiles]
        self.coef_, self.pvalue_ = zip(*models_)
        return self

    def predict(self, X: np.array):
        """
        Predict / estimate quantiles
        """
        X = self._array_to_df(X)
        preds = [X.assign(const=1).multiply(c).sum(axis=1) for c in self.coef_]
        preds_df = pd.concat(preds, axis=1)
        preds_df.columns = self.prediction_cols
        return preds_df

    def score(self, X: np.array, y: np.array):
        """
        Default score function. Returns the tilted loss
        """
        y_pred = self.predict(X)
        scores = [
            tilted_loss(y_true=y, y_pred=y_pred[f"predict_q_{q}"], quantile=q)
            for q in self.quantiles
        ]
        score = np.mean(scores)
        return score
