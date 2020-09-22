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
        quantile: float, default=0.5
            Quantile to fit, with `` 0 < quantile < 1 ``.
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
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    >>>                 variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    >>>                            'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    >>>
    >>> # Fit model
    >>> model = LinearQuantileRegression(quantile=0.99)
    >>> model.fit(X_train, y_train)
    """
    def __init__(self, quantile=0.5, tol=1e-3, max_iter=1000):
        self.quantile = quantile
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """ Fit a Linear Quantile Regression using statsmodels
        """
        # Statsmodels requires to add constant columns manually
        # otherwise the models will not fit an intercept
        self.model_ = QuantReg(
            y,
            smapi.add_constant(X)
        )
        self.model_result_ = self.model_.fit(
            q=self.quantile,
            p_tol=self.tol,
            max_iter=self.max_iter
        )
        return self

    def predict(self, X):
        """ Predict / estimate quantiles
        """
        return self.model_result_.predict(
            smapi.add_constant(X)
        )

    def score(self, X, y):
        """ Default score Function. Returns the tilted loss
        """
        y_pred = self.predict(X)
        score = tilted_loss(y_true=y, y_pred=y_pred, quantile=self.quantile)
        return score
