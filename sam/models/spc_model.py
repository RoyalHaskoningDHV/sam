from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sam.models.base_model import BaseTimeseriesRegressor
from sam.utils.sklearnhelpers import FunctionTransformerWithNames
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SPCTemplate(BaseEstimator, RegressorMixin):
    """SPC template for usage in BaseQuantileRegressor

    This template class follows scikit-learn estimator principles and
    always predicts quantiles and median values.

    During predict it uses the number of rows in the input to determine the
    shape of the prediction. Apart from that, the model only uses the target y
    for all calculations.

    Parameters
    ----------
    quantiles: Sequence of float, optional
        The quantiles that need to be present in the output, by default ()
    predict_ahead: Sequence of int, optional
        How many timepoints we want to predict ahead. Is only used to determine the
        number of rows in the output, by default (1,)
    """

    def __init__(
        self, quantiles: Sequence[float] = (), predict_ahead: Sequence[int] = (1,)
    ) -> None:
        self.quantiles = quantiles
        self.predict_ahead = predict_ahead

    def fit(self, X: Any, y: Any, **kwargs: dict):
        """Fit the SPC model

        The X parameter is only used so it is compatible with sklearn.
        Fits the quantiles and median values using the y data and saves
        to self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data, not used in this function. This parameter is only
            used for compatability.
        y : array-like of shape (n_samples,)
            The target data that is used for determining the median and quantile
            values

        Returns
        -------
        The fitted model
        """
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        self.spc_median_ = np.median(y)
        self.spc_quantiles_ = np.quantile(y[~np.isnan(y)], q=np.sort(self.quantiles))
        return self

    def predict(self, X: Any):
        """Predict using the SPC model

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

        prediction = np.append(self.spc_quantiles_, self.spc_median_)
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


class SPCRegressor(BaseTimeseriesRegressor):
    """SPC Regressor model

    Baseline model that always predict the median and quantiles.
    This model can be used as a benchmark or fall-back method, since the
    predicted median and quantiles can still be used to trigger alarms.
    Also see https://en.wikipedia.org/wiki/Statistical_process_control

    This model uses the same init parameters as the other SAM models for
    compatability, but ignores all of the feature engineering parameters

    Note: using use_diff_of_y changes how this model works; instead of predicting
    static bounds, it will fit the median and quantiles on the differenced target
    then it will undo the differencing by adding those values to the last timestep,
    resulting in a model that predicts the last timestep + the median difference.
    This approach works especially when trying to predict a signal that has a continuous
    trend.

    Parameters
    ----------
    predict_ahead: integer, optional (default=(1,))
        how many steps to predict ahead. For example, if (1, 2), the model will predict both 1 and
        2 timesteps into the future. If (0), predict the present. If not equal to (0),
        predict the future, with differencing.
        A single integer is also allowed, in which case the value is converted to a singleton list.
    quantiles: array-like, optional (default=())
        The quantiles to predict. Between 0 and 1. Keep in mind that the median will be predicted
        regardless of this parameter
    use_y_as_feature: boolean, optional (default=False)
        Not used in this class, only for compatibility.
    use_diff_of_y: bool, optional (default=True)
        If True differencing is used (the difference between y now and shifted y),
        else differencing is not used (shifted y is used).
    timecol: string, optional (default=None)
        If not None, the column to use for constructing time features. For now,
        creating features from a DateTimeIndex is not supported yet.
    y_scaler: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform and inverse_transform method.
        E.g.: StandardScaler() or PowerTransformer()
    time_components: array-like, optional (default=('minute', 'hour', 'day', 'weekday'))
        Not used in this class, only for compatibility.
    time_cyclicals: array-like, optional (default=('minute', 'hour', 'day'))
        Not used in this class, only for compatibility.
    time_onehots: array-like, optional (default=('weekday'))
        Not used in this class, only for compatibility.
    rolling_window_size: array-like, optional (default=(5,))
        Not used in this class, only for compatibility.
    rolling_features: array-like, optional (default=('mean'))
        Not used in this class, only for compatibility.

    Examples
    --------
    >>> from sam.models import SPCRegressor
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
    >>> model = SPCRegressor(timecol='TIME', quantiles=(0.25, 0.75))
    ...
    >>> model.fit(X_train, y_train)
    >>> pred = model.predict(X_test, y_test)
    >>> pred.head()
           predict_lead_1_q_0.25  predict_lead_1_q_0.75  predict_lead_1_mean
    11655                   56.0                  158.0                101.0
    11656                   56.0                  158.0                101.0
    11657                   56.0                  158.0                101.0
    11658                   56.0                  158.0                101.0
    11659                   56.0                  158.0                101.0
    """

    def __init__(
        self,
        predict_ahead: int = (1,),
        quantiles: Sequence[float] = (),
        use_y_as_feature: bool = False,
        use_diff_of_y: bool = False,
        timecol: str = None,
        y_scaler: TransformerMixin = None,
        time_components: Sequence[str] = None,
        time_cyclicals: Sequence[str] = None,
        time_onehots: Sequence[str] = None,
        rolling_window_size: Sequence[int] = (),
        rolling_features: Sequence[str] = None,
    ) -> None:
        self.predict_ahead = predict_ahead
        self.quantiles = quantiles
        self.use_y_as_feature = use_y_as_feature
        self.use_diff_of_y = use_diff_of_y
        self.timecol = timecol
        self.y_scaler = y_scaler
        self.time_components = time_components
        self.time_cyclicals = time_cyclicals
        self.time_onehots = time_onehots
        self.rolling_features = rolling_features
        self.rolling_window_size = rolling_window_size

    def get_feature_engineer(self) -> Pipeline:
        """
        The SPC model doesn't do any feature building, but it has to impute values
        since sam can't work with NaNs and pass the column names in a transformer
        called 'columns'.

        Returns
        -------
        sklearn.pipeline.Pipeline:
            The feature building pipeline
        """
        columns = self._input_cols
        if self.timecol:
            columns = [col for col in columns if col != self.timecol]

        feature_engineering_steps = [
            ("passthrough", FunctionTransformerWithNames(validate=False), columns),
        ]
        engineer = ColumnTransformer(feature_engineering_steps, remainder="drop")
        return Pipeline([("columns", engineer), ("impute", SimpleImputer())])

    def get_untrained_model(self) -> Callable:
        """Returns an underlying model that can be trained

        Creates an instance of the SPCTemplate class

        Returns
        -------
        A trainable model class
        """
        return SPCTemplate(self.quantiles, self.predict_ahead)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        **fit_kwargs,
    ) -> Callable:
        """Fit the SPCRegressor model

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
        Predict using the SPCRegressor

        This will either predict the static bounds that were fitted during
        fit() or when using `use_diff_of_y` it will predict the last timestep plus
        the median/quantile difference.

        In the first situation X is only used to determine how many datapoints
        need to be predicted. In the latter case it will use X to undo the differencing.

        For compatibility reasons the method acceps predict_kwargs, that are not used.

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
        A fitted SPCRegressor object
        """
        import cloudpickle

        foldername = Path(foldername)
        with open(foldername / (prefix + ".pkl"), "rb") as f:
            obj = cloudpickle.load(f)

        return obj
