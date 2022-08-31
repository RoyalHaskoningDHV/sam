from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sam.models.base_model import BaseTimeseriesRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ConstantTemplate(BaseEstimator, RegressorMixin):
    """Constant regression template for usage in BaseQuantileRegressor

    This template class follows scikit-learn estimator principles and
    always predicts quantiles and median values.

    During predict it uses the number of rows in the input to determine the
    shape of the prediction. Apart from that, the model only uses the target y
    for all calculations.

    Parameters
    ----------
    predict_ahead: Sequence of int or int, optional (default=0)
        How many timepoints we want to predict ahead. Is only used to determine the
        number of rows in the output.
    quantiles: Sequence of float, optional (default=())
        The quantiles that need to be present in the output.
    average_type: str, optional (default="median")
        The type of average that is used to calculate the median and quantiles.

    """

    def __init__(
        self,
        predict_ahead: Sequence[int] = (0,),
        quantiles: Sequence[float] = (),
        average_type: str = "median",
    ) -> None:
        self.predict_ahead = predict_ahead
        self.quantiles = quantiles
        self.average_type = average_type

    def fit(self, X: Any, y: Any, **kwargs: dict):
        """Fit the model

        The X parameter is only used so it is compatible with sklearn.
        Fits the quantiles and median values using the y data and saves
        to self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data, not used in this function. This parameter is only
            used for compatibility.
        y : array-like of shape (n_samples,)
            The target data that is used for determining the median and quantile
            values

        Returns
        -------
        The fitted model
        """
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        if self.average_type == "median":
            self.model_average_ = np.median(y)
        else:
            self.model_average_ = np.mean(y)
        self.model_quantiles_ = np.quantile(y[~np.isnan(y)], q=np.sort(self.quantiles))
        return self

    def predict(self, X: Any):
        """Predict using the model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            input data, only used to determine the amount of rows that need
            to be present in the output.

        Returns
        -------
        array-like of shape (n_samples * predict_ahead, len(quantiles) + 1)
            output of the prediction, containing the median and and quantiles
        """
        check_is_fitted(self)
        X = check_array(X)

        prediction = np.append(self.model_quantiles_, self.model_average_)
        prediction = np.repeat(prediction, len(self.predict_ahead))
        if len(self.quantiles) > 1:
            return np.tile(prediction, (len(X), 1))
        else:
            return np.tile(prediction, len(X))

    def _more_tags(self):
        """helper function to make sure this class
        passes the check_estimator check
        """
        return {"poor_score": True}


class ConstantTimeseriesRegressor(BaseTimeseriesRegressor):
    """Constant Regression model

    Baseline model that always predict the median and quantiles.
    This model can be used as a benchmark or fall-back method, since the
    predicted median and quantiles can still be used to trigger alarms.
    Also see https://en.wikipedia.org/wiki/Statistical_process_control

    This model uses the same init parameters as the other SAM models for
    compatibility, but ignores all of the feature engineering parameters

    Note: using use_diff_of_y changes how this model works; instead of predicting
    static bounds, it will fit the median and quantiles on the differenced target
    then it will undo the differencing by adding those values to the last timestep,
    resulting in a model that predicts the last timestep + the median difference.
    This approach works especially when trying to predict a signal that has a continuous
    trend.

    Parameters
    ----------
    predict_ahead: tuple of integers, optional (default=(0,))
        how many steps to predict ahead. For example, if (1, 2), the model will predict both 1 and
        2 timesteps into the future. If (0,), predict the present. If not equal to (0,),
        predict the future. Combine with `use_diff_of_y` to get a persistence benchmark forecasting
        model.
    quantiles: tuple of floats, optional (default=())
        The quantiles to predict. Values between 0 and 1. Keep in mind that the mean will be
        predicted regardless of this parameter
    use_diff_of_y: bool, optional (default=True)
        If True differencing is used (the difference between y now and shifted y),
        else differencing is not used (shifted y is used).
    timecol: string, optional (default=None)
        If not None, the column to use for constructing time features. For now,
        creating features from a DateTimeIndex is not supported yet.
    y_scaler: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform and inverse_transform method.
        E.g.: StandardScaler() or PowerTransformer().
    average_type: str = "median",
        The type of average that is used to calculate the median and quantiles.
        FOR NOW ONLY "median" IS SUPPORTED.
    kwargs: dict, optional
        Not used. Just for compatibility with other SAM models.

    Examples
    --------
    >>> from sam.models import ConstantTimeseriesRegressor
    >>> from sam.data_sources import read_knmi
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    ...
    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    ...                  variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    ...                             'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    ...
    >>> model = ConstantTimeseriesRegressor(timecol='TIME', quantiles=[0.25, 0.75])
    ...
    >>> model.fit(X_train, y_train)
    >>> pred = model.predict(X_test, y_test)
    >>> pred.head()
           predict_lead_0_q_0.25  predict_lead_0_q_0.75  predict_lead_0_mean
    11655                   56.0                  158.0                101.0
    11656                   56.0                  158.0                101.0
    11657                   56.0                  158.0                101.0
    11658                   56.0                  158.0                101.0
    11659                   56.0                  158.0                101.0
    """

    def __init__(
        self,
        predict_ahead: Sequence[int] = (0,),
        quantiles: Sequence[float] = (),
        use_diff_of_y: bool = False,
        timecol: str = None,
        y_scaler: TransformerMixin = None,
        average_type: str = "median",
        **kwargs,
    ) -> None:
        super().__init__(
            predict_ahead=predict_ahead,
            quantiles=quantiles,
            use_diff_of_y=use_diff_of_y,
            timecol=timecol,
            y_scaler=y_scaler,
            average_type=average_type,
            **kwargs,
        )

    def get_untrained_model(self) -> Callable:
        """Returns an underlying model that can be trained

        Creates an instance of the ConstantTemplate class

        Returns
        -------
        A trainable model class
        """
        return ConstantTemplate(predict_ahead=self.predict_ahead, quantiles=self.quantiles)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        **fit_kwargs,
    ) -> Callable:
        """Fit the ConstantTimeseriesRegressor model

        This function will preprocess the input data, get the untrained underlying model
        and fits the model.

        For compatibility reasons the method acceps fit_kwargs, that are not used.

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to 'train' the model
        y: pd.Series
            Target data (dependent variable) used to 'train' the model.
        validation_data: tuple(pd.DataFrame, pd.Series) (X_val, y_val respectively)
            Data used for validation step

        Returns
        -------
        Always returns None, since there is no history object of the fit procedure
        """
        X_transformed, y_transformed, _, _ = self.preprocess_fit(X, y, validation_data)
        self.model_ = self.get_untrained_model()
        self.model_.fit(X_transformed, y_transformed)
        return None

    def predict(
        self, X: pd.DataFrame, y: pd.Series = None, return_data: bool = False, **predict_kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Predict using the ConstantTimeseriesRegressor

        This will either predict the static bounds that were fitted during
        fit() or when using `use_diff_of_y` it will predict the last timestep plus
        the median/quantile difference.

        In the first situation X is only used to determine how many datapoints
        need to be predicted. In the latter case it will use X to undo the differencing.

        For compatibility reasons the method accepts predict_kwargs, that are not used.

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to predict.
        y: pd.Series
            The target values
        return_data: bool, optional (default=False)
            whether to return only the prediction, or to return both the prediction and the
            transformed input (X) dataframe.

        Returns
        -------
        prediction: pd.DataFrame
            The predictions coming from the model
        X_transformed: pd.DataFrame, optional
            The transformed input data, when return_data is True, otherwise None
        """
        self.validate_data(X)

        X_transformed = self.preprocess_predict(X, y)
        prediction = self.model_.predict(X_transformed)

        prediction = self.postprocess_predict(prediction, X, y)

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def dump(self, foldername: str, prefix: str = "model") -> None:
        """
        Writes the instanced model to foldername/prefix.pkl

        prefix is configurable, and is 'model' by default

        Overwrites the abstract method from SamQuantileRegressor

        Parameters
        ----------
        foldername: str
            The name of the folder to save the model
        prefix: str, optional (Default='model')
            The name of the model
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, "model_")

        import cloudpickle

        foldername = Path(foldername)

        with open(foldername / (prefix + ".pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, foldername, prefix="model") -> Callable:
        """
        Reads and loads the model located at foldername/prefix.pkl

        prefix is configurable, and is 'model' by default
        Output is an entire instance of the fitted model that was saved

        Overwrites the abstract method from SamQuantileRegressor

        Returns
        -------
        A fitted ConstantTimeseriesRegressor object
        """
        import cloudpickle

        foldername = Path(foldername)
        with open(foldername / (prefix + ".pkl"), "rb") as f:
            obj = cloudpickle.load(f)

        return obj
