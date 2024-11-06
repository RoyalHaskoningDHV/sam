from pathlib import Path
from typing import Callable, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
from sam.feature_engineering import BaseFeatureEngineer
from sam.models import BaseTimeseriesRegressor
from sklearn.base import TransformerMixin
from sklearn.linear_model import Lasso, QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor


class LassoTimeseriesRegressor(BaseTimeseriesRegressor):
    """Linear quantile regression model (Lasso) for time series

    This model combines several approaches to time series data:
    Multiple outputs for forecasting, quantile regression, and feature engineering.
    This class implements a linear quantile regression model. For each quantile, a
    separate model is trained.

    It is a wrapper around
    the sklearn `QuantileRegressor` and `Lasso`. For more information on the
    model specifics, see the sklearn documentation.

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
    average_type: str (default='mean')
        Determines what to fit as the average: 'mean', or 'median'. The average is the last
        node in the output layer and does not reflect a quantile, but rather estimates the central
        tendency of the data. Setting to 'mean' results in fitting that node with MSE, and
        setting this to 'median' results in fitting that node with MAE (equal to 0.5 quantile).
    feature_engineering: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform method, e.g.
        `sam.feature_engineering.SimpleFeatureEngineer`.
    alpha : float, default=1.0
        Regularization constant that multiplies the L1 penalty term.
    fit_intercept : bool, default=True
        Whether or not to fit the intercept.
    quantile_options : dict, optional (default=None)
        Options for `sklearn.linear_model.QuantileRegressor`.
    mean_options : dict, optional (default=None)
        Options for `sklearn.linear_model.Lasso`.
    kwargs: dict, optional
        Not used. Just for compatibility of models that inherit from this class.

    Attributes
    ----------
    feature_engineer_: Sklearn transformer
        The transformer used on the raw data before prediction
    n_inputs_: integer
        The number of inputs used for the underlying neural network
    n_outputs_: integer
        The number of outputs (columns) from the model
    prediction_cols_: array of strings
        The names of the output columns from the model.
    model_ : object
        List of sklearn models, one for each quantile.

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.models import MLPTimeseriesRegressor
    >>> from sam.feature_engineering import SimpleFeatureEngineer
    >>> from sam.datasets import load_rainbow_beach
    ...
    >>> data = load_rainbow_beach()
    >>> X, y = data, data["water_temperature"]

    >>> simple_features = SimpleFeatureEngineer(keep_original=False)
    >>> model = MLPTimeseriesRegressor(
    ...     predict_ahead=(0,),
    ...     feature_engineer=simple_features,
    ...     verbose=0,
    ... )
    >>> model.fit(X, y)  # doctest: +ELLIPSIS
    <keras.src.callbacks.history.History ...
    """

    def __init__(
        self,
        predict_ahead: Sequence[int] = (0,),
        quantiles: Sequence[float] = (),
        use_diff_of_y: bool = False,
        timecol: str = None,
        y_scaler: TransformerMixin = None,
        average_type: str = "mean",
        feature_engineer: BaseFeatureEngineer = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        quantile_options: dict = None,
        mean_options: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            predict_ahead=predict_ahead,
            quantiles=quantiles,
            use_diff_of_y=use_diff_of_y,
            timecol=timecol,
            y_scaler=y_scaler,
            feature_engineer=feature_engineer,
            **kwargs,
        )
        self.average_type = average_type
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.quantile_options = quantile_options
        self.mean_options = mean_options
        self.feature_engineer = feature_engineer

        if self.average_type == "median" and 0.5 in self.quantiles:
            raise ValueError(
                "average_type is mean, but 0.5 is also in quantiles (duplicate). "
                "Either set average_type to mean or remove 0.5 from quantiles"
            )

    def get_untrained_model(self, quantile=None) -> Callable:
        """Returns linear quantile regression model"""
        if quantile is not None:
            estimator = QuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                solver="highs",
                fit_intercept=self.fit_intercept,
                **(self.quantile_options or {}),
            )
        else:
            estimator = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                **(self.mean_options or {}),
            )
        return MultiOutputRegressor(estimator=estimator)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **fit_kwargs,
    ):
        X, y, _, _ = self.preprocess_fit(X, y)
        self.model_ = [self.get_untrained_model(quantile) for quantile in self.quantiles]

        if self.average_type == "mean":
            self.model_.append(self.get_untrained_model())
        elif self.average_type == "median":
            self.model_.append(self.get_untrained_model(0.5))
        else:
            raise ValueError(f"Unknown average_type: {self.average_type}")

        for model in self.model_:
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

        if y is None and self.use_diff_of_y:
            raise ValueError("You must provide y when using use_diff_of_y=True")

        X_transformed = self.preprocess_predict(X, y)
        prediction = [model.predict(X_transformed) for model in self.model_]
        prediction = np.concatenate(prediction, axis=1)

        prediction = self.postprocess_predict(
            prediction, X, y, force_monotonic_quantiles=force_monotonic_quantiles
        )

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def dump_parameters(self, foldername: str, prefix: str = "model") -> None:
        import cloudpickle

        with open(Path(foldername) / f"{prefix}_params.pkl", "wb") as f:
            cloudpickle.dump(self.model_, f)

    @staticmethod
    def load_parameters(obj, foldername: str, prefix: str = "model") -> Any:
        import cloudpickle

        with open(Path(foldername) / f"{prefix}_params.pkl", "rb") as f:
            model = cloudpickle.load(f)
        return model
