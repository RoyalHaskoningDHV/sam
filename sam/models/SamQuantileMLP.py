import warnings
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
from sam.feature_engineering import BuildRollingFeatures, decompose_datetime
from sam.metrics import R2Evaluation, keras_joint_mse_tilted_loss
from sam.models import create_keras_quantile_mlp
from sam.preprocessing import inverse_differenced_target, make_shifted_target
from sam.utils import FunctionTransformerWithNames, assert_contains_nans
from sklearn import __version__ as skversion
from sklearn.base import BaseEstimator
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
            preprocess_before_predict = SamQuantileMLP.preprocess_before_predict

        self.model = SamProxyModel()
        # Trick sklearn into thinking this is a fitted variable
        self.model.feature_engineer_ = model.feature_engineer_
        # Will likely be somewhere around 0
        self.expected_value = explainer.expected_value

    def shap_values(
        self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs
    ) -> np.array:
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
        X_transformed = self.model.preprocess_before_predict(X, y, dropna=False)
        return self.explainer.shap_values(X_transformed, *args, **kwargs)

    def attributions(
        self, X: pd.DataFrame, y: pd.Series = None, *args, **kwargs
    ) -> np.array:
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
        X_transformed = self.model.preprocess_before_predict(X, y, dropna=False)
        return pd.DataFrame(
            X_transformed, columns=self.model.feature_names_, index=X.index
        )


class SamQuantileMLP(BaseEstimator):
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
    predict_ahead: integer, optional (default=[1])
        how many steps to predict ahead. For example, if [1, 2], the model will predict both 1 and
        2 timesteps into the future. If [0], predict the present. If not equal to [0],
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
    lr: integer, optional (default=0.001)
        The learning rate to use in the model, see :ref:`create_keras_quantile_mlp
        <create-keras-quantile-mlp>`
    dropout: integer, optional (default=None)
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
    >>>
    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    >>>                 variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    >>>                            'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    >>>
    >>> # We are predicting the weather 1 hour ahead. Since weather is highly autocorrelated, we
    >>> # expect this the persistence benchmark to score decently high, but also it should be
    >>> # easy to predict the weather 1 hour ahead, so the model should do even better.
    >>> model = SamQuantileMLP(predict_ahead=1, use_y_as_feature=True, timecol='TIME',
    >>>                       quantiles=[0.25, 0.75], epochs=5,
    >>>                       time_components=['hour', 'month', 'weekday'],
    >>>                       time_cyclicals=['hour', 'month', 'weekday'],
    >>>                       rolling_window_size=[1,5,6])
    >>>
    >>> # fit returns a keras history callback object, which can be used as normally
    >>> history = model.fit(X_train, y_train)
    >>> pred = model.predict(X_test, y_test).dropna()
    >>>
    >>> actual = model.get_actual(y_test).dropna()
    >>> # Because of impossible to know values, some rows have to be dropped. After dropping
    >>> # them, make sure the indexes still match by dropping the same rows from each side
    >>> pred, actual = pred.reindex(actual.index).dropna(), actual.reindex(pred.index).dropna()
    >>> mean_squared_error(actual, pred.iloc[:, -1])  # last column contains mean prediction
    114.50628975834859
    >>>
    >>> # Persistence corresponds to predicting the present, so use ytest
    >>> persistence_prediction = y_test.reindex(actual.index).dropna()
    >>> mean_squared_error(actual, persistence_prediction)
    149.35018919848642
    >>>
    >>> # As we can see, the model performs significantly better than the persistence benchmark
    >>> # Mean benchmark, which does much worse:
    >>> mean_prediction = pd.Series(y_test.mean(), index = actual)
    >>> mean_squared_error(actual, mean_prediction)
    2410.20138157309
    """

    def __init__(
        self,
        predict_ahead=1,
        quantiles=(),
        use_y_as_feature=True,
        use_diff_of_y=True,
        timecol=None,
        y_scaler=None,
        time_components=["minute", "hour", "day", "weekday"],
        time_cyclicals=["minute", "hour", "day"],
        time_onehots=["weekday"],
        rolling_window_size=(12,),
        rolling_features=["mean"],
        n_neurons=200,
        n_layers=2,
        batch_size=16,
        epochs=20,
        lr=0.001,
        dropout=None,
        momentum=None,
        verbose=1,
        r2_callback_report=False,
        average_type="mean",
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

        if (self.average_type == "median" and 0.5 in self.quantiles):
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

        return Pipeline(
            [("columns", engineer), ("impute", imputer), ("scaler", scaler)]
        )

    def get_untrained_model(self) -> Callable:
        """
        Returns a simple 2d keras model.
        This is just a wrapper for sam.models.create_keras_quantile_mlp
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

    def validate_predict_ahead(self):
        """
        Perform checks to validate the predict_ahead attribute
        """
        if np.isscalar(self.predict_ahead):
            self.predict_ahead = [self.predict_ahead]

        if not all([p >= 0 for p in self.predict_ahead]):
            raise ValueError("All values of predict_ahead must be 0 or larger!")

        if self.predict_ahead == [0] and self.use_diff_of_y:
            raise ValueError("use_diff_of_y must be false when predict_ahead is 0")

        if self.predict_ahead == [0] and self.use_y_as_feature:
            raise ValueError("use_y_as_feature must be false when predict_ahead is 0")

        if len(np.unique(self.predict_ahead)) != len(self.predict_ahead):
            raise ValueError("predict_ahead contains double values")

    def validate_data(self, X: pd.DataFrame) -> None:
        """
        Validates the data and raises an exception if:
        - There is no time columns
        - The data is not monospaced
        - There is not enought data

        Parameters
        ----------
        x: pd.DataFrame
            The dataframe to validate
        """
        if self.timecol is None:
            warnings.warn(
                (
                    "No timecolumn given. Make sure the data is"
                    "monospaced when given to this model!"
                ),
                UserWarning,
            )
        else:
            monospaced = X[self.timecol].diff()[1:].unique().size == 1
            if not monospaced:
                raise ValueError(
                    "Data is not monospaced, which is required for"
                    "this model. fit/predict is not possible"
                )

        enough_data = len(self.rolling_window_size) == 0 or X.shape[0] > max(
            self.rolling_window_size
        )
        if not enough_data:
            warnings.warn(
                "Not enough data given to calculate rolling features. "
                "Output will be entirely missing values.",
                UserWarning,
            )

    def prepare_input_and_target(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the input dataframe X and target series y by:
        - Scaling y
        - Adding target as feature to input
        - Transforming the target

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to 'train' the model
        y: pd.Series
            Target data (dependent variable) used to 'train' the model

        Returns
        -------
        X: pd.DataFrame:
            The (enriched) training input dataframe
        y_transformed: pd.Series
            The transformed target series
        """
        if self.y_scaler is not None:
            y = pd.Series(
                self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel(),
                index=y.index,
                name=y.name,
            )

        if self.use_y_as_feature:
            X = X.assign(y_=y.copy())

        # Create the actual target
        if self.predict_ahead != [0]:
            y_transformed = make_shifted_target(
                y, self.use_diff_of_y, self.predict_ahead
            )
        else:
            # Dataframe with 1 column. Will use y's index and name
            y_transformed = pd.DataFrame(y.copy()).astype(float)

        return X, y_transformed

    @staticmethod
    def verify_same_indexes(X: pd.DataFrame, y: pd.Series, y_can_be_none=True):
        """
        Verify that X and y have the same index
        """
        if not y.index.equals(X.index):
            raise ValueError("For training, X and y must have an identical index")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        **fit_kwargs
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

        SamQuantileMLP.verify_same_indexes(X, y)
        self.validate_predict_ahead()
        self.validate_data(X)

        # Update the input and target depending on settings
        X, y_transformed = self.prepare_input_and_target(X, y)

        # Index where target is nan, cannot be trained on.
        targetnanrows = y_transformed.isna().any(axis=1)

        # Save input columns names
        self._set_input_cols(X)

        # buildrollingfeatures
        self.rolling_cols_ = [col for col in X if col != self.timecol]
        self.feature_engineer_ = self.get_feature_engineer()

        # Apply feature engineering
        X_transformed = self.feature_engineer_.fit_transform(X)

        # Now we have fitted the feature engineer, we can set the feature names
        self.set_feature_names(X, X_transformed)

        # Now feature names are set, we can start using self.get_feature_names()
        X_transformed = pd.DataFrame(
            X_transformed, columns=self.get_feature_names(), index=X.index
        )

        self.n_inputs_ = len(self.get_feature_names())

        # Create output column names. In this model, our outputs are assumed to have the
        # form: [quantile_1_output_1, quantile_1_output_2, ... ,
        # quantile_n_output_1, quantile_n_output_2, ..., mean_output_1, mean_output_2]
        # Where n_output (1 or 2 in this example) is decided by self.predict_ahead
        self.prediction_cols_ = [
            "predict_lead_{}_q_{}".format(p, q)
            for q in self.quantiles
            for p in self.predict_ahead
        ]
        self.prediction_cols_ += [
            "predict_lead_{}_mean".format(p) for p in self.predict_ahead
        ]
        self.n_outputs_ = len(self.prediction_cols_)

        # Remove the first n rows because they are nan anyway because of rolling features
        if len(self.rolling_window_size) > 0:
            X_transformed = X_transformed.iloc[max(self.rolling_window_size) :]
            y_transformed = y_transformed.iloc[max(self.rolling_window_size) :]
        # Filter rows where the target is unknown
        X_transformed = X_transformed.loc[~targetnanrows]
        y_transformed = y_transformed.loc[~targetnanrows]

        assert_contains_nans(
            X_transformed, "Data cannot contain nans. Imputation not supported for now"
        )

        self.model_ = self.get_untrained_model()

        # Apply transformations to validation data if provided:
        if validation_data is not None:
            X_val, y_val = validation_data

            self.validate_data(X_val)

            if self.y_scaler is not None:
                y_val = pd.Series(
                    self.y_scaler.transform(y_val.values.reshape(-1, 1)).ravel(),
                    index=y_val.index,
                    name=y_val.name,
                )

            # This does not affect training: it only calls transform, not fit
            X_val_transformed = self.preprocess_before_predict(X_val, y_val)
            X_val_transformed = pd.DataFrame(
                X_val_transformed, columns=self.get_feature_names(), index=X_val.index
            )
            if self.predict_ahead != [0]:
                y_val_transformed = make_shifted_target(
                    y_val, self.use_diff_of_y, self.predict_ahead
                )
            else:
                # Dataframe with 1 column. Will use y's index and name
                y_val_transformed = pd.DataFrame(y_val.copy()).astype(float)

            # Remove the first n rows because they are nan anyway because of rolling features
            if len(self.rolling_window_size) > 0:
                X_val_transformed = X_val_transformed.iloc[
                    max(self.rolling_window_size) :
                ]
                y_val_transformed = y_val_transformed.iloc[
                    max(self.rolling_window_size) :
                ]
            # The lines below are only to deal with nans in the validation set
            # These should eventually be replaced by Arjans/Fennos function for removing nan rows
            # So that this code will be much more readable
            targetnanrows = y_val_transformed.isna().any(axis=1)

            # Filter rows where the target is unknown
            X_val_transformed = X_val_transformed.loc[~targetnanrows]
            y_val_transformed = y_val_transformed.loc[~targetnanrows]
            # Until here
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
            **fit_kwargs
        )
        return history

    def preprocess_before_predict(
        self, X: pd.DataFrame, y: pd.Series, dropna: bool = False
    ) -> pd.DataFrame:
        """
        Transform a DataFrame X so it can be fed to self.model_.
        This is useful for several usecases, where you want to use the underlying
        keras model as opposed to the wrapper. For example shap, eli5, and even just
        implementing the `predict` function.

        Parameters
        ----------
        X: pd.DataFrame
            The training input samples.
        y: pd.Series
            The target values
        dropna: bool, optional (default=False)
            If True, delete the rows that contain NaN values
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, "feature_engineer_")

        if y is not None:
            SamQuantileMLP.verify_same_indexes(X, y)

        if self.use_y_as_feature:
            X = X.assign(y_=y.copy())

        X_transformed = self.feature_engineer_.transform(X)
        if dropna:
            X_transformed = X_transformed[~np.isnan(X_transformed).any(axis=1)]
        return X_transformed

    def predict(
        self, X: pd.DataFrame, y: pd.Series = None, return_data: bool = False
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

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to predict.
        y: pd.Series
            The target values
        return_data: bool, optional (default=False)
            whether to return only the prediction, or to return both the prediction and the
            transformed input (X) dataframe.
        """
        if self.predict_ahead != 0 and y is None:
            raise ValueError("When predict_ahead > 0, y is needed for prediction")

        if y is not None:
            SamQuantileMLP.verify_same_indexes(X, y)

        self.validate_data(X)

        X_transformed = self.preprocess_before_predict(X, y)
        prediction = self.model_.predict(X_transformed)

        # Put the predictions in a dataframe so we can undo the differencing
        prediction = pd.DataFrame(
            prediction, columns=self.prediction_cols_, index=X.index
        )
        # If we performed differencing, we undo it here
        if self.use_diff_of_y:
            prediction = inverse_differenced_target(prediction, y)

        if prediction.shape[1] == 1:
            # If you just wanted to predict a single value without quantiles,
            # just return a series. Easier to work with.
            prediction = prediction.iloc[:, 0]

            if self.y_scaler is not None:
                prediction = pd.Series(
                    self.y_scaler.inverse_transform(prediction.values).ravel(),
                    index=prediction.index,
                    name=prediction.name,
                )

        else:
            if self.y_scaler is not None:
                inv_pred = np.zeros_like(prediction)
                for i in range(prediction.shape[1]):
                    inv_pred[:, i] = self.y_scaler.inverse_transform(
                        prediction.iloc[:, i].values.reshape(-1, 1)
                    ).ravel()
                prediction = pd.DataFrame(
                    inv_pred, columns=prediction.columns, index=prediction.index
                )

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def set_feature_names(self, X: Any, X_transformed: Any) -> None:
        """
        Default function for setting self._feature_names

        For the default feature engineer, it outputs features like 'mean__Q#mean_1'
        Which is much easier to interpret if we remove the 'mean__'

        Parameters
        ----------
        X: Unused
        X_transformed: Unused
        """
        names = self.feature_engineer_.named_steps["columns"].get_feature_names()
        names = [colname.split("__")[1] for colname in names]
        self._feature_names = names

    def get_feature_names(self) -> list:
        """
        Function for obtaining feature names. Generally used instead of the attribute, and more
        compatible with the sklearn API.

        Returns
        -------
        list:
            list of feature names
        """
        check_is_fitted(self, "_feature_names")
        return self._feature_names

    def _set_input_cols(self, X: pd.DataFrame) -> None:
        """
        Function to set the attribute self._input_cols (input column names).
        Only used internally right before the feature building.
        Time column is not included, since time is always a dependency
        This can be used to determine model dependencies

        Parameters
        ----------
        X: pd.DataFrame
            The DataFrame that contains the input columns
        """
        col_names = X.columns.values
        col_names = col_names[col_names != self.timecol]
        self._input_cols = col_names

    def get_input_cols(self) -> np.array:
        """
        Function to obtain the input column names.
        This can be used to determine model dependencies
        Time column is not included, since time is always a dependency

        Returns
        -------
        list:
            The input column names
        """
        check_is_fitted(self, "_input_cols")
        return self._input_cols

    def get_actual(self, y: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """
        Convenience function for getting the actual values (perfect prediction).
        Mainly useful for scoring the model. This essentially does and undoes differencing
        on y, meaning this function will output what a perfect model would have outputted.
        If predict_ahead is 0, no differencing is done anyway, so y is just returned unchanged.

        Returns a Series of a Dataframe depending on the number of values in self.predict_ahead

        Parameters
        -------
        y:
            The target values

        Returns
        -------
        pd.Series or pd.DataFrame:
            y after applying differencing and undoing the process (if self.use_diff_of_y)
            If self.predict_ahead is a single value, this function will return a series.
            If self.predict_ahead has multiple values (list), this function will return a
            dataframe.
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, "model_")
        if len(self.predict_ahead) == 1:
            pred = self.predict_ahead[0]
        else:
            pred = self.predict_ahead

        if self.predict_ahead != [0]:
            actual = make_shifted_target(y, self.use_diff_of_y, pred)
            if self.use_diff_of_y:
                actual = inverse_differenced_target(actual, y)
        else:
            actual = y.copy().astype(float)

        return actual

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Default score function. Uses sum of mse and tilted loss

        Parameters
        ----------
        X: pd.DataFrame
            The independent variables used to predict.
        y: pd.Series
            The target values

        Returns:
        float:
            The score
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, "model_")
        # We need a dataframe, regardless of if these functions outputs a series or dataframe
        prediction = pd.DataFrame(self.predict(X, y))
        actual = pd.DataFrame(self.get_actual(y))

        # scale these predictions back to get a score that is in same units as keras loss
        if self.y_scaler is not None:
            pred_scaled = np.zeros_like(prediction)
            for i in range(prediction.shape[1]):
                pred_scaled[:, i] = self.y_scaler.transform(
                    prediction.iloc[:, i].values.reshape(-1, 1)
                ).ravel()
            prediction = pd.DataFrame(
                pred_scaled, columns=prediction.columns, index=prediction.index
            )

            actual = pd.DataFrame(
                self.y_scaler.transform(actual.values).ravel(),
                index=actual.index,
                columns=actual.columns,
            )

        # actual usually has some missings at the end
        # prediction usually has some missings at the beginning
        # We ignore the rows with missings
        missings = actual.isna().any(axis=1) | prediction.isna().any(axis=1)
        actual = actual.loc[~missings]
        prediction = prediction.loc[~missings]

        # self.prediction_cols_[-1] defines the mean prediction
        # For n outputs, we need the last n columns instead
        # Therefore, this line calculates the mse of the mean prediction
        mean_prediction = prediction[
            self.prediction_cols_[-1 * len(self.predict_ahead) :]
        ].values
        # Calculate the MSE of all the predictions, and tilted loss of quantile predictions,
        # then sum them at the end
        loss = np.sum(np.mean((actual - mean_prediction) ** 2, axis=0))
        for i, q in enumerate(self.quantiles):
            startcol = len(self.predict_ahead) * i
            endcol = startcol + len(self.predict_ahead)
            e = np.array(
                actual - prediction[self.prediction_cols_[startcol:endcol]].values
            )
            # Calculate all the quantile losses, and sum them at the end
            loss += np.sum(np.mean(np.max([q * e, (q - 1) * e], axis=0), axis=0))
        return loss

    def dump(self, foldername: str, prefix: str = "model") -> None:
        """
        Writes the following files:
        * prefix.pkl
        * prefix.h5

        to the folder given by foldername. prefix is configurable, and is
        'model' by default

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

        Returns
        -------
        Keras model
        """
        import cloudpickle
        from tensorflow.keras.models import load_model

        foldername = Path(foldername)
        with open(foldername / (prefix + ".pkl"), "rb") as f:
            obj = cloudpickle.load(f)

        loss = obj._get_loss()
        obj.model_ = load_model(
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
                loss = keras_joint_mse_tilted_loss(
                    y, f, self.quantiles, len(self.predict_ahead)
                )
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

        X_transformed = self.preprocess_before_predict(X, y)

        if self.predict_ahead != [0]:
            y_target = make_shifted_target(
                y, self.use_diff_of_y, self.predict_ahead
            ).iloc[:, 0]

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

        X_transformed = self.preprocess_before_predict(X, y, dropna=True)
        if sample_n is not None:
            # Sample some rows to increase performance later
            sampled = np.random.choice(X_transformed.shape[0], sample_n, replace=False)
            X_transformed = X_transformed[sampled, :]
        explainer = shap.DeepExplainer(self.model_, X_transformed)
        return SamShapExplainer(explainer, self)
