from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sam.feature_engineering import BuildRollingFeatures, decompose_datetime
from sam.metrics import R2Evaluation, keras_joint_mse_tilted_loss
from sam.models import create_keras_quantile_mlp
from sam.models.base_model import BaseTimeseriesRegressor
from sam.preprocessing import make_shifted_target
from sam.utils import FunctionTransformerWithNames
from sklearn import __version__ as skversion
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class SamShapExplainer(object):
    def __init__(self, explainer: Callable, model: Callable) -> None:
        """
        An object that imitates a SHAP explainer object. (Sort of) implements the base Explainer
        interface which can be found here
        <https://github.com/slundberg/shap/blob/master/shap/explainers/explainer.py>.
        The more advanced, tensorflow-specific attributes can be accessed with obj.explainer.
        The reason the interface is only sort of implemented, is the same reason why SamQuantileMLP
        doesn't entirely implement the skearn interface - for predicting, y is needed, which is
        not supported by the SamShapExplainer.

        Parameters
        ----------
        explainer: shap TFDeepExplainer object
            A shap explainer object. This will be used to generate the actual shap values
        model: SAMQuantileMLP model
            This will be used to do the preprocessing before calling explainer.shap_values
        """
        self.explainer = explainer

        # Create a proxy model that can call only 3 attributes we need
        class SamProxyModel:
            fit = None
            use_y_as_feature = model.use_y_as_feature
            feature_names_ = model.get_feature_names()
            preprocess_predict = SamQuantileMLP.preprocess_predict

        self.model = SamProxyModel()
        # Trick sklearn into thinking this is a fitted variable
        self.model.feature_engineer_ = model.feature_engineer_
        # Will likely be somewhere around 0
        self.expected_value = explainer.expected_value

    def shap_values(self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs) -> np.array:
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

    def attributions(self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs) -> np.array:
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


class SamQuantileMLP(BaseTimeseriesRegressor):
    """
    This is an example class for how the SAM skeleton can work. This is not the final/only model,
    there are some notes:
    - There is no validation yet. Therefore, the input data must already be sorted and monospaced
    - The feature engineering is very simple, we just calculate lag/max/min/mean for a given window
      size, as well as minute/hour/month/weekday if there is a time column
    - The prediction requires y as input. The reason for this is described in the predict function.
      Keep in mind that this is not directly 'cheating', since we are predicting a future value of
      y, and giving the present value of y as input to the predict.
      When predicting the present, this y is not needed and can be None

    It is possible to subclass this class and overwrite functions. For now, the most obvious case
    is overwriting the `get_feature_engineer(self)` function. This function must return a
    transformer, with attributes: `fit`, `transform`, `fit_transform`, and `get_feature_names`.
    The output of `transform` must be a numpy array or pandas dataframe with the same number of
    rows as the input array. The output of `get_feature_names` must be an array or list of
    strings, the same length as the number of columns in the output of `transform`.

    Another possibility would be overwriting the `get_untrained_model(self)` function.
    This function must return a keras model, with `fit`, `predict`, `save` and `summary`
    attributes, where fit/predict will accept a regular (2d) dataframe or numpy array as input.

    Note that the below parameters are just for the default model, and subclasses can have
    different `__init__` parameters.

    Parameters
    ----------
    predict_ahead: integer or list of integers, optional (default=1)
        how many steps to predict ahead. For example, if [1, 2], the model will predict both 1 and
        2 timesteps into the future. If [0], predict the present. If not equal to 0 or [0],
        predict the future, with differencing.
        A single integer is also allowed, in which case the value is converted to a singleton list.
    quantiles: array-like, optional (default=())
        The quantiles to predict. Between 0 and 1. Keep in mind that the mean will be predicted
        regardless of this parameter
    use_y_as_feature: boolean, optional (default=True)
        Whether or not to use y as a feature for predicting. If predict_ahead=0, this must be False
        Due to time limitations, for now, this option has to be True, unless predict_ahead is 0.
        Potentially in the future, this option can also be False when predict_ahead is not 0.s
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
        The timefeatures to create. See :ref:`decompose_datetime`.
    time_cyclicals: array-like, optional (default=('minute', 'hour', 'day'))
        The cyclical timefeatures to create. See :ref:`decompose    _datetime`.
    time_onehots: array-like, optional (default=('weekday'))
        The onehot timefeatures to create. See :ref:`decompose_datetime`.
    rolling_window_size: array-like, optional (default=(5,))
        The window size to use for `BuildRollingFeatures`
    rolling_features: array-like, optional (default=('mean'))
        The rolling functions to generate in the default feature engineering function.
        Values should be interpretable by BuildRollingFeatures.
    n_neurons: integer, optional (default=200)
        The number of neurons to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    n_layers: integer, optional (default=2)
        The number of layers to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    batch_size: integer, optional (default=16)
        The batch size to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    epochs: integer, optional (default=20)
        The number of epochs to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    lr: float, optional (default=0.001)
        The learning rate to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    dropout: float, optional (default=None)
        The type of dropout to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    momentum: integer, optional (default=None)
        The type of momentum in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    verbose: integer, optional (default=1)
        The verbosity of fitting the keras model. Can be either 0, 1 or 2.
    r2_callback_report: boolean (default=False)
        Whether to add train and validation r2 to each epoch as a callback.
        This also changes self.verbose to 2 to prevent log print mess up.
    average_type: str (default='mean')
        Determines what to fit as the average: 'mean', or 'median'. The average is the last
        node in the output layer and does not reflect a quantile, but rather estimates the central
        tendency of the data. Setting to 'mean' results in fitting that node with MSE, and
        setting this to 'median' results in fitting that node with MAE (equal to 0.5 quantile).

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
    model_: Keras model
        The underlying keras model

    Examples
    --------
    >>> from sam.models import SamQuantileMLP
    >>> from sam.data_sources import read_knmi
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    >>> import tensorflow as tf
    >>> tf.random.set_seed(42)
    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    ...                  variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    ...                             'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    >>> # We are predicting the weather 1 hour ahead. Since weather is highly autocorrelated, we
    >>> # expect this the persistence benchmark to score decently high, but also it should be
    >>> # easy to predict the weather 1 hour ahead, so the model should do even better.
    >>> model = SamQuantileMLP(predict_ahead=1, use_y_as_feature=True, timecol='TIME',
    ...                        quantiles=(0.25, 0.75), epochs=5, verbose=0,
    ...                        time_components=('hour', 'month', 'weekday'),
    ...                        time_cyclicals=('hour', 'month', 'weekday'),
    ...                        time_onehots=None,
    ...                        rolling_window_size=(1,5,6))
    >>> # fit returns a keras history callback object, which can be used as normally
    >>> history = model.fit(X_train, y_train)
    >>> pred = model.predict(X_test, y_test).dropna()
    >>> actual = model.get_actual(y_test).dropna()
    >>> # Because of impossible to know values, some rows have to be dropped. After dropping
    >>> # them, make sure the indexes still match by dropping the same rows from each side
    >>> pred, actual = pred.reindex(actual.index).dropna(), actual.reindex(pred.index).dropna()
    >>> mse_score = mean_squared_error(actual, pred.iloc[:, -1])  # last column contains mean preds
    >>> print(round(mse_score, 2))
    66.08
    >>> # Persistence corresponds to predicting the present, so use ytest
    >>> persistence_prediction = y_test.reindex(actual.index).dropna()
    >>> persistence_mse_score = mean_squared_error(actual, persistence_prediction)
    >>> print(round(persistence_mse_score, 2))
    149.45
    >>> # As we can see, the model performs significantly better than the persistence benchmark
    >>> # Mean benchmark, which does much worse:
    >>> mean_prediction = pd.Series(y_test.mean(), index = actual)
    >>> bench_mse_score = mean_squared_error(actual, mean_prediction)
    >>> print(round(bench_mse_score, 2))
    2410.59
    """

    def __init__(
        self,
        predict_ahead: Union[int, Sequence[int]] = 1,
        quantiles: Sequence[float] = (),
        use_y_as_feature: bool = True,
        use_diff_of_y: bool = True,
        timecol: str = None,
        y_scaler: TransformerMixin = None,
        time_components: Sequence[str] = ("minute", "hour", "day", "weekday"),
        time_cyclicals: Sequence[str] = ("minute", "hour", "day"),
        time_onehots: Sequence[str] = ("weekday",),
        rolling_window_size: Sequence[int] = (12,),
        rolling_features: Sequence[str] = ("mean",),
        n_neurons: int = 200,
        n_layers: int = 2,
        batch_size: int = 16,
        epochs: int = 20,
        lr: float = 0.001,
        dropout: float = None,
        momentum: float = None,
        verbose: int = 1,
        r2_callback_report: bool = False,
        average_type: str = "mean",
    ) -> None:
        super().__init__(
            predict_ahead=predict_ahead,
            quantiles=quantiles,
            use_y_as_feature=use_y_as_feature,
            use_diff_of_y=use_diff_of_y,
            timecol=timecol,
            y_scaler=y_scaler,
            time_components=time_components,
            time_cyclicals=time_cyclicals,
            time_onehots=time_onehots,
            rolling_features=rolling_features,
            rolling_window_size=rolling_window_size,
        )
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.dropout = dropout
        self.verbose = verbose
        self.r2_callback_report = r2_callback_report
        self.average_type = average_type

        if self.average_type == "median" and 0.5 in self.quantiles:
            raise ValueError(
                "average_type is mean, but 0.5 is also in quantiles. "
                "Either set average_type to mean or add 0.5 to quantiles"
            )

    def get_feature_engineer(self) -> Pipeline:
        """
        Function that returns a sklearn Pipeline with a column transformer, an imputer and
        a scaler

        The steps followed by the columntransformer are:
        - On the time col (if it was passed), does decompose_datetime and cyclicals
        - On the other columns, calculates lag/max/min/mean features for a given window size
        - On the other (nontime) columns, passes them through unchanged (same as lag 0)

        The imputer is passed in case some datapoints are missing.
        The scaler is used to improve the MLP performance

        Overwrites the abstract method from SamQuantileRegressor

        Returns
        -------
        sklearn.pipeline.Pipeline:
            The pipeline with steps 'columns', 'impute', 'scaler'
        """

        def time_transformer(dates: pd.DataFrame) -> pd.DataFrame:
            return decompose_datetime(
                dates,
                self.timecol,
                components=self.time_components,
                cyclicals=self.time_cyclicals,
                onehots=self.time_onehots,
                keep_original=False,
            )

        def identity(x):
            return x

        feature_engineer_steps = [
            # Rolling features
            (
                rol,
                BuildRollingFeatures(
                    rolling_type=rol,
                    window_size=self.rolling_window_size,
                    lookback=0,
                    keep_original=False,
                ),
                self.rolling_cols_,
            )
            for rol in self.rolling_features
        ] + [
            # Other features
            (
                "passthrough",
                FunctionTransformerWithNames(identity, validate=False),
                self.rolling_cols_,
            )
        ]
        if self.timecol:
            feature_engineer_steps += [
                (
                    "timefeats",
                    FunctionTransformerWithNames(time_transformer, validate=False),
                    [self.timecol],
                )
            ]

        # Drop the time column if exists
        engineer = ColumnTransformer(feature_engineer_steps, remainder="drop")

        imputer = SimpleImputer()
        scaler = StandardScaler()

        return Pipeline([("columns", engineer), ("impute", imputer), ("scaler", scaler)])

    def get_untrained_model(self) -> Callable:
        """
        Returns a simple 2d keras model.
        This is just a wrapper for sam.models.create_keras_quantile_mlp

        Overwrites the abstract method from SamQuantileRegressor
        """

        return create_keras_quantile_mlp(
            n_input=self.n_inputs_,
            n_neurons=self.n_neurons,
            n_layers=self.n_layers,
            n_target=len(self.predict_ahead),
            quantiles=self.quantiles,
            lr=self.lr,
            momentum=self.momentum,
            dropout=self.dropout,
            average_type=self.average_type,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        **fit_kwargs,
    ) -> Callable:
        """
        This function does the following:
        - Validate that the input is monospaced and has enough rows
        - Perform differencing on the target
        - Create feature engineer by calling `self.get_feature_engineer()`
        - Fitting/applying the feature engineer
        - Bookkeeping to create the output columns
        - Remove rows with nan that can't be used for fitting
        - Get untrained model with `self.get_untrained_model()`
        - Fit the untrained model and return the history object
        - Optionally, preprocess validation data to give to the fit
        - Pass through any other fit_kwargs to the fit function

        Overwrites the abstract method from SamQuantileRegressor

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
        tf.keras.callbacks.History:
            The history object after fitting the keras model
        """
        (
            X_transformed,
            y_transformed,
            X_val_transformed,
            y_val_transformed,
        ) = self.preprocess_fit(X, y, validation_data)

        self.model_ = self.get_untrained_model()

        if validation_data is not None:
            validation_data = (X_val_transformed, y_val_transformed)

        if self.r2_callback_report:

            all_data = {"X_train": X_transformed, "y_train": y_transformed}

            if validation_data is not None:
                all_data["X_val"] = X_val_transformed
                all_data["y_val"] = y_val_transformed

            # append to the callbacks argument
            if "callbacks" in fit_kwargs.keys():
                # early stopping should be last callback to work properly
                fit_kwargs["callbacks"] = [
                    R2Evaluation(all_data, self.prediction_cols_, self.predict_ahead)
                ] + fit_kwargs["callbacks"]
            else:
                fit_kwargs["callbacks"] = [
                    R2Evaluation(all_data, self.prediction_cols_, self.predict_ahead)
                ]

            # Keras verbosity can be in [0, 1, 2].
            # If verbose is 1, this means that the user wants to display messages.
            # However, keras printing messes up with custom callbacks and verbose == 1,
            # see: https://github.com/keras-team/keras/issues/2354
            # The only solution for now is to set verbose to 2 when using custom callbacks.
            if self.verbose == 1:
                self.verbose = 2

        # Fit model
        history = self.model_.fit(
            X_transformed,
            y_transformed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=validation_data,
            **fit_kwargs,
        )
        return history

    def predict(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        return_data: bool = False,
        force_monotonic_quantiles: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Make a prediction, and undo differencing in the case it was used

        Important! This is different from sklearn/tensorflow API...
        We need y during prediction for two reasons:
        1) a lagged version is used for feature engineering
        2) The underlying model can predict a differenced number, and then we want to output the
           'real' prediction, so we need y to undo the differencing
        Keep in mind that prediction will work if you are predicting the future. e.g. you have
        data from 00:00-12:00, and are predicting 4 hours into the future, it will predict what
        the value will be at 4:00-16:00

        Overwrites the abstract method from SamQuantileRegressor

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to predict.
        y: pd.Series
            The target values
        return_data: bool, optional (default=False)
            whether to return only the prediction, or to return both the prediction and the
            transformed input (X) dataframe.
        force_monotonic_quantiles: bool, optional (default=False)
            whether to force quantiles to not overlap. When fitting multiple quantile regressions
            it is possible that individual quantile regression lines over-lap, or in other words,
            a quantile regression line fitted to a lower quantile predicts higher that a line
            fitted to a higher quantile. If this occurs for a certain prediction, the output
            distribution is invalid. We can force monotonicity by making the outer quantiles
            at least as high as the inner quantiles.

        Returns
        -------
        prediction: pd.DataFrame
            The predictions coming from the model
        X_transformed: pd.DataFrame, optional
            The transformed input data, when return_data is True, otherwise None
        """
        if max(self.predict_ahead) > 0 and y is None:
            raise ValueError("When predict_ahead > 0, y is needed for prediction")

        self.validate_data(X)

        X_transformed = self.preprocess_predict(X, y)
        prediction = self.model_.predict(X_transformed)

        prediction = self.postprocess_predict(
            prediction, X, y, force_monotonic_quantiles=force_monotonic_quantiles
        )

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def dump(self, foldername: str, prefix: str = "model") -> None:
        """
        Writes the following files:
        * prefix.pkl
        * prefix.h5

        to the folder given by foldername. prefix is configurable, and is
        'model' by default

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

        # TEMPORARY
        self.model_.save(foldername / (prefix + ".h5"))

        # Set the models to None temporarily, because they can't be pickled
        backup, self.model_ = self.model_, None

        with open(foldername / (prefix + ".pkl"), "wb") as f:
            cloudpickle.dump(self, f)

        # Set it back
        self.model_ = backup

    @classmethod
    def load(cls, foldername, prefix="model") -> Callable:
        """
        Reads the following files:
        * prefix.pkl
        * prefix.h5

        from the folder given by foldername. prefix is configurable, and is
        'model' by default
        Output is an entire instance of the fitted model that was saved

        Overwrites the abstract method from SamQuantileRegressor

        Returns
        -------
        Keras model
        """
        import cloudpickle
        from tensorflow import keras

        foldername = Path(foldername)
        with open(foldername / (prefix + ".pkl"), "rb") as f:
            obj = cloudpickle.load(f)

        loss = obj._get_loss()
        obj.model_ = keras.models.load_model(
            foldername / (prefix + ".h5"), custom_objects={"mse_tilted": loss}
        )
        return obj

    def _get_loss(self) -> Union[str, Callable]:
        """
        Convenience function, mirrors create_keras_quantile_mlp
        Only needed for loading, since it is a custom object, it is not
        saved in the .h5 file by default
        """
        if len(self.quantiles) == 0:
            mse_tilted = "mse"
        else:

            def mse_tilted(y, f):
                loss = keras_joint_mse_tilted_loss(y, f, self.quantiles, len(self.predict_ahead))
                return loss

        return mse_tilted

    def summary(self, print_fn: Callable = print) -> None:
        """
        Combines several methods to create a 'wrapper' summary method.

        Parameters
        ----------
        print_fn: Callable (default=print)
            A function for writting down results
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, "model_")
        print_fn(str(self))
        print_fn(self.get_feature_names())
        self.model_.summary(print_fn=print_fn)

    def quantile_feature_importances(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        score: Union[str, Callable] = None,
        n_iter: int = 5,
        sum_time_components: bool = False,
    ) -> pd.DataFrame:
        """
        Computes feature importances based on the loss function used to estimate the average.
        This function uses ELI5's `get_score_importances:
        <https://eli5.readthedocs.io/en/latest/autodocs/permutation_importance.html>`
        to compute feature importances. It is a method that measures how the score decreases when
        a feature is not available.
        This is essentially a model-agnostic type of feature importance that works with
        every model, including keras MLP models.

        Note that we compute feature importance over the average (the central trace, and the last
        output node, either median or mean depending on self.average_type), and do not include
        the quantiles in the loss calculation. Initially, the quantiles were included, but
        experimentation showed that importances behaved very badly when including the
        quantiles in the loss: importances were sometimes consistently negative (i.e. in all
        random iterations), while these features should have been important according to theory,
        and excluding them indeed lead to much worse model performance. This behavior goes away
        when only using the mean trace to estimate feature importance.

        Parameters
        ----------
        X: pd.DataFrame
            dataframe with test or train features
        y: pd.Series
            dataframe with test or train target
        score: str or function, optional (default=None)
            Either a function with signature score(X, y, model)
            that returns a scalar. Will be used to measure score decreases for ELI5.
            If None, defaults to MSE or MAE depending on self.average_type.
            Note that if score computes a loss (i.e. higher is worse), negative values indicate
            positive contribution to model performance (i.e. negative score decrease means that
            removing this feature will increase the metric, which is a bad thing with MAE/MSE).
        n_iter: int, optional (default=5)
            Number of iterations to use for ELI5. Since ELI5 results can vary wildly, increasing
            this parameter may provide more stablitity at the cost of a longer runtime
        sum_time_components: bool, optional (default=False)
            if set to true, sums feature importances of the different subfeatures of each time
            component (i.e. weekday_1, weekday_2 etc. in one 'weekday' importance)

        Returns
        -------
        score_decreases: Pandas dataframe,  shape (n_iter x n_features)
            The score decreases when leaving out each feature per iteration. The larget the
            magnitude, the more important each feature is considered by the model.

        Examples
        --------
        >>> # Example with a fictional dataset with only 2 features
        >>> score_decreases = \
        >>>     model.quantile_feature_importances(X_test[:100], y_test[:100], n_iter=3)
        >>> # The score decreases of each feature in each iteration
        >>> score_decreases
            feature_1 feature_2
        0   5.5       4.3
        1   5.1       2.3
        2   5.0       2.4

        >>> feature_importances = score_decreases.mean()
        feature_1    5.2
        feature_2    3.0
        dtype: float64

        >>> # This will show a barplot of all the score importances, with error bars
        >>> seaborn.barplot(data=score_decreases)
        """
        # Model must be fitted for this method
        check_is_fitted(self, "model_")

        if len(self.predict_ahead) > 1:
            raise NotImplementedError(
                "This method is currently not implemented " "for multiple targets"
            )

        if int(skversion.split(".")[1]) >= 24:
            raise Exception("This feature requires sklearn version < 0.24.0!")

        from eli5.permutation_importance import get_score_importances

        if score is None:

            def score(X, y, model=self.model_):
                if self.average_type == "median":
                    return np.mean(np.abs(y - model.predict(X)[:, -1]))
                elif self.average_type == "mean":
                    return np.mean((y - model.predict(X)[:, -1]) ** 2)

        X_transformed = self.preprocess_predict(X, y)

        if self.predict_ahead != [0]:
            y_target = make_shifted_target(y, self.use_diff_of_y, self.predict_ahead).iloc[:, 0]

        else:
            y_target = y.copy().astype(float)

        # Remove rows with missings in either of the two arrays
        missings = np.isnan(y_target) | np.isnan(X_transformed).any(axis=1)
        X_transformed = X_transformed[~missings, :]
        y_target = y_target[~missings]

        # use eli5 to compute feature importances:
        base_scores, score_decreases = get_score_importances(
            score, X_transformed, y_target, n_iter=n_iter
        )

        decreases_df = pd.DataFrame(score_decreases, columns=self.get_feature_names())

        if sum_time_components:
            for component in self.time_components:
                these_cols = [
                    c
                    for c in decreases_df.columns
                    if c.startswith("%s_%s_" % (self.timecol, component))
                ]
                decreases_df[component] = decreases_df[these_cols].sum(axis=1)
                decreases_df = decreases_df.drop(these_cols, axis=1)

        return decreases_df

    def get_explainer(
        self, X: pd.DataFrame, y: pd.Series = None, sample_n: int = None
    ) -> SamShapExplainer:
        """
        Obtain a shap explainer-like object. This object can be used to
        create shap values and explain predictions.

        Keep in mind that this will explain
        the created features from `self.get_feature_names()`, not the input features.
        To help with this, the explainer comes with a `test_values()` attribute
        that calculates the test values corresponding to the shap values

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe used to 'train' the explainer
        y: pd.Series, optional (default=None)
            Target data used to 'train' the explainer. Only required when self.predict_ahead > 0.
        sample_n: integer, optional (default=None)
            The number of samples to give to the explainer. It is reccommended that
            if your background set is greater than 5000, to sample for performance reasons.

        Returns
        -------
        SamShapExplainer:
            Custom Sam object that inherits from shap.DeepExplainer

        Examples
        --------
        >>> explainer = model.get_explainer(X_test, y_test, sample_n=1000)
        >>> shap_values = explainer.shap_values(X_test[0:10], y_test[0:10])
        >>> test_values = explainer.test_values(X_test[0:10], y_test[0:10])

        >>> shap.force_plot(explainer.expected_value[0], shap_values[0][-1,:],
        >>>                 test_values.iloc[-1,:], matplotlib=True)
        """
        import shap

        X_transformed = self.preprocess_predict(X, y, dropna=True)
        if sample_n is not None:
            # Sample some rows to increase performance later
            sampled = np.random.choice(X_transformed.shape[0], sample_n, replace=False)
            X_transformed = X_transformed[sampled, :]
        explainer = shap.DeepExplainer(self.model_, X_transformed)
        return SamShapExplainer(explainer, self)
