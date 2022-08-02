import logging
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sam.feature_engineering import BaseFeatureEngineer
from sam.models import BaseTimeseriesRegressor
from sklearn.base import TransformerMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor


class LassoTimeseriesRegressor(BaseTimeseriesRegressor):
    """
    
    Parameters
    ----------
    predict_ahead: tuple of integers, optional (default=(0,))
        how many steps to predict ahead. For example, if (1, 2), the model will predict both 1 and
        2 timesteps into the future. If (0,), predict the present.
    quantiles: tuple of floats, optional (default=())
        The quantiles to predict. Values between 0 and 1. Keep in mind that the mean will be
        predicted regardless of this parameter
    use_diff_of_y: bool, optional (default=False)
        If True differencing is used (the difference between y now and shifted y),
        else differencing is not used (shifted y is used).
    timecol: string, optional (default=None)
        If not None, the column to use for constructing time features. For now,
        creating features from a DateTimeIndex is not supported yet.
    y_scaler: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform and inverse_transform method.
        E.g.: StandardScaler() or PowerTransformer()
    fit_mean: bool, optional (default=False)
        If True, regular linear regression is used to fit the mean in addition to the
        quantiles.
    feature_engineering: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform method, e.g.
        `sam.feature_engineering.SimpleFeatureEngineer`.
    alpha : float, default=1.0
        Regularization constant that multiplies the L1 penalty term.
    fit_intercept : bool, default=True
        Whether or not to fit the intercept.
    solver : {'highs-ds', 'highs-ipm', 'highs', 'interior-point', \
            'revised simplex'}, default='interior-point'
        Method used by :func:`scipy.optimize.linprog` to solve the linear
        programming formulation. Note that the highs methods are recommended
        for usage with `scipy>=1.6.0` because they are the fastest ones.
        Solvers "highs-ds", "highs-ipm" and "highs" support
        sparse input data and, in fact, always convert to sparse csc.
    solver_options : dict, default=None
        Additional parameters passed to :func:`scipy.optimize.linprog` as
        options. If `None` and if `solver='interior-point'`, then
        `{"lstsq": True}` is passed to :func:`scipy.optimize.linprog` for the
        sake of stability.
    kwargs: dict, optional
        Not used. Just for compatibility of models that inherit from this class.

    """

    def __init__(
        self,
        predict_ahead: Sequence[int] = (0,),
        quantiles: Sequence[float] = (),
        use_diff_of_y: bool = False,
        timecol: str = None,
        y_scaler: TransformerMixin = None,
        fit_mean: bool = False,
        feature_engineer: BaseFeatureEngineer = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "interior-point",
        solver_options: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            predict_ahead=predict_ahead,
            quantiles=quantiles,
            use_diff_of_y=use_diff_of_y,
            timecol=timecol,
            y_scaler=y_scaler,
            fit_mean=fit_mean,
            feature_engineer=feature_engineer,
            **kwargs,
        )
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_options = solver_options

    def get_untrained_model(self, quantile) -> Callable:
        """Returns linear quantile regression model"""
        model = MultiOutputRegressor(
            estimator=QuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                solver_options=self.solver_options,
            )
        )
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **fit_kwargs,
    ):
        X, y, _, _ = self.preprocess_fit(X, y)
        self.models_ = [self.get_untrained_model(quantile) for quantile in self.quantiles]
        for quantile, model in zip(self.quantiles, self.models_):
            logging.info(f"Fitting model for quantile {quantile}")
            model.fit(X, y, **fit_kwargs)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        return_data: bool = False,
        force_monotonic_quantiles: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        self.validate_data(X)
        X_transformed = self.preprocess_predict(X, y)
        predictions = []
        for quantile, model in zip(self.quantiles, self.models_):
            logging.info(f"Predicting quantile {quantile}")
            predictions.append(model.predict(X_transformed))
        prediction = np.concatenate(predictions, axis=1)

        prediction = self.postprocess_predict(
            prediction, X, y, force_monotonic_quantiles=force_monotonic_quantiles
        )

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def dump(self, foldername: str, prefix: str = "model") -> None:
        """Save a model to disk

        This abstract method needs to be implemented by any class inheriting from
        SamQuantileRegressor. This function dumps the SAM model to disk.

        Parameters
        ----------
        foldername : str
            The folder location where to save the model
        prefix : str, optional
           The prefix used in the filename, by default "model"
        """
        return None

    @classmethod
    def load(cls, foldername, prefix="model") -> Callable:
        """Load a model from disk

        This abstract method needs to be implemented by any class inheriting from
        SamQuantileRegressor. This function loads a SAM model from disk.

        Parameters
        ----------
        foldername : str
            The folder location where the model is stored
        prefix : str, optional
           The prefix used in the filename, by default "model"

        Returns
        -------
        The SAM model that has been loaded from disk
        """
        return None
