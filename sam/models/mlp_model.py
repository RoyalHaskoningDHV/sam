from pathlib import Path
from typing import Callable, Sequence, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
from keras import Optimizer, Model
from sklearn.pipeline import Pipeline

from sam.feature_engineering import BaseFeatureEngineer
from sam.metrics import R2Evaluation, keras_joint_mse_tilted_loss
from sam.models import create_keras_quantile_mlp
from sam.models.base_model import BaseTimeseriesRegressor
from sam.preprocessing import make_shifted_target
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sam.models.sam_shap_explainer import SamShapExplainer
from sam.utils.score_importance import get_score_importances


class MLPTimeseriesRegressor(BaseTimeseriesRegressor):
    """Multi-layer Perceptron Regressor for time series

    This model combines several approaches to time series data:
    Multiple outputs for forecasting, quantile regression, and feature engineering.
    This class is an implementation of an MLP to estimate multiple quantiles for all
    forecasting horizons at once.

    This is a wrapper for a keras MLP model. For more information on the model parameters,
    see the keras documentation.

    Parameters
    ----------
    predict_ahead: tuple of integers, optional (default=(0,))
        how many steps to predict ahead. For example, if (1, 2), the model will predict both 1 and
        2 timesteps into the future. If (0,), predict the present.
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
        E.g.: StandardScaler() or PowerTransformer()
    feature_engineering: object, optional (default=None)
        Should be an sklearn-type transformer that has a transform method, e.g.
        `sam.feature_engineering.SimpleFeatureEngineer`.
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
    optimizer: Optimizer (default=None)
        Forcefully overwrites the default Adam optimizer object.

    kwargs: dict, optional
        Not used. Just for compatibility with other SAM models.

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
    >>> import pandas as pd
    >>> from sam.models import MLPTimeseriesRegressor
    >>> from sam.feature_engineering import SimpleFeatureEngineer
    >>> from sam.datasets import load_rainbow_beach
    ...
    >>> data = load_rainbow_beach()
    >>> X, y = data, data["water_temperature"]

    >>> simple_features = SimpleFeatureEngineer(
    ...     rolling_features=[
    ...         ("wave_height", "mean", 24),
    ...         ("wave_height", "mean", 12),
    ...     ],
    ...     time_features=[
    ...         ("hour_of_day", "cyclical"),
    ...     ],
    ...     keep_original=False,
    ... )
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
        feature_engineer: Union[BaseFeatureEngineer, Pipeline] = None,
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
        optimizer: Optional[Optimizer] = None,
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
        self.optimizer = optimizer

        self.input_shape = None

        if self.average_type == "median" and 0.5 in self.quantiles:
            raise ValueError(
                "average_type is mean, but 0.5 is also in quantiles (duplicate). "
                "Either set average_type to mean or remove 0.5 from quantiles"
            )

        self.to_save_objects = ["feature_engineer_", "y_scaler"]
        self.to_save_parameters = ["prediction_cols_", "quantiles", "predict_ahead"]

    def get_untrained_model(self) -> Callable:
        """
        Returns a simple 2d keras model.
        This is just a wrapper for sam.models.create_keras_quantile_mlp

        Overwrites the abstract method from BaseTimeseriesRegressor
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
            optimizer=self.optimizer,
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

        Overwrites the abstract method from BaseTimeseriesRegressor

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
        tf.keras.src.callbacks.history.History:
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
            r2_eval_callback = R2Evaluation(all_data, self.prediction_cols_, self.predict_ahead)
            if "callbacks" in fit_kwargs.keys():
                # early stopping should be last callback to work properly
                fit_kwargs["callbacks"] = [r2_eval_callback] + fit_kwargs["callbacks"]
            else:
                fit_kwargs["callbacks"] = [r2_eval_callback]

            # Keras verbosity can be in [0, 1, 2].
            # If verbose is 1, this means that the user wants to display messages.
            # However, keras printing messes up with custom callbacks and verbose == 1,
            # see: https://github.com/keras-team/keras/issues/2354
            # The only solution for now is to set verbose to 2 when using custom callbacks.
            if self.verbose == 1:
                self.verbose = 2

        # Set input shape for ONNX
        self.input_shape = X_transformed.values.shape[1:]

        # Fit model
        history = self.model_.fit(
            X_transformed.values,
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

        Overwrites the abstract method from BaseTimeseriesRegressor

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
        import onnxruntime as ort

        self.validate_data(X)

        if y is None and self.use_diff_of_y:
            raise ValueError("You must provide y when using use_diff_of_y=True")

        X_transformed = self.preprocess_predict(X, y)

        prediction = None
        if isinstance(self.model_, ort.InferenceSession):
            prediction = self.model_.run([], {"X": X_transformed.values.astype(np.float32)})[0]

        if isinstance(self.model_, Model):
            prediction = self.model_.predict(X_transformed, verbose=self.verbose)

        prediction = self.postprocess_predict(
            prediction, X, y, force_monotonic_quantiles=force_monotonic_quantiles
        )

        if return_data:
            return prediction, X_transformed
        else:
            return prediction

    def dump_parameters(
        self, foldername: str, prefix: str = "model", file_extension=".h5"
    ) -> None:
        """
        Writes the following files:
        * prefix.h5

        to the folder given by foldername. prefix is configurable, and is
        'model' by default

        Overwrites the abstract method from BaseTimeseriesRegressor

        Parameters
        ----------
        foldername: str
            The name of the folder to save the model
        prefix: str, optional (Default='model')
            The name of the model
        file_extension: str, optional (default=".h5")
            What file extension to use.
        """
        import tf2onnx
        import onnx
        import tensorflow as tf

        check_is_fitted(self, "model_")
        foldername = Path(foldername)
        if file_extension == ".onnx":
            input_signature = [tf.TensorSpec((None, *self.input_shape), name="X")]
            onnx_model, _ = tf2onnx.convert.from_keras(
                self.model_, input_signature=input_signature, opset=13
            )
            onnx.save(onnx_model, foldername / (prefix + ".onnx"))
            return
        elif file_extension == ".h5":
            self.model_.save(foldername / (prefix + ".h5"))
            return

        raise ValueError(
            f"The file extension: {file_extension} " f"is not supported, choose '.onnx' or '.h5'"
        )

    @staticmethod
    def load_parameters(obj, foldername: str, prefix: str = "model") -> Any:
        """
        Loads the file:
        * prefix.h5

        from the folder given by foldername. prefix is configurable, and is
        'model' by default
        Output is the `model_` attribute of the MLPTimeseriesRegressor class.

        Overwrites the abstract method from BaseTimeseriesRegressor
        """
        import keras
        import os
        import onnxruntime as ort

        foldername = Path(foldername)
        loss = obj._get_loss()
        file_path = foldername / prefix
        if os.path.exists(file_path := file_path.with_suffix(".onnx")):
            return ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
        if os.path.exists(file_path := file_path.with_suffix(".h5")):
            return keras.models.load_model(file_path, custom_objects={"mse_tilted": loss})
        raise FileNotFoundError(f"Could not find parameter file: {prefix}.onnx or {prefix}.h5")

    def _get_loss(self) -> Union[str, Callable]:
        """
        Convenience function, mirrors create_keras_quantile_mlp
        Only needed for loading, since it is a custom object, it is not
        saved in the .h5 file by default
        """
        if len(self.quantiles) == 0:
            return "mse"
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
        self.model_.summary(print_fn=print_fn)

    def quantile_feature_importances(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        score: Union[str, Callable] = None,
        n_iter: int = 5,
        sum_time_components: bool = False,
        random_state: int = None,
    ) -> pd.DataFrame:
        """
        Computes feature importances based on the loss function used to estimate the average.
        This function uses an adaptation of ELI5's `get_score_importances:
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
            this parameter may provide more stability at the cost of a longer runtime
        sum_time_components: bool, optional (default=False)
            if set to true, sums feature importances of the different subfeatures of each time
            component (i.e. weekday_1, weekday_2 etc. in one 'weekday' importance)
        random_state: int, optional (default=None)
            Used for shuffling columns of matrix columns.

        Returns
        -------
        score_decreases: Pandas dataframe,  shape (n_iter x n_features)
            The score decreases when leaving out each feature per iteration. The larger the
            magnitude, the more important each feature is considered by the model.

        Examples
        --------
        >>> # Example with a fictional dataset with only 2 features
        >>> import pandas as pd
        >>> import seaborn
        >>> from sam.models import MLPTimeseriesRegressor
        >>> from sam.feature_engineering import SimpleFeatureEngineer
        >>> from sam.datasets import load_rainbow_beach
        ...
        >>> data = load_rainbow_beach()
        >>> X, y = data, data["water_temperature"]
        >>> test_size = int(X.shape[0] * 0.33)
        >>> train_size = X.shape[0] - test_size
        >>> X_train, y_train = X.iloc[:train_size, :], y[:train_size]
        >>> X_test, y_test = X.iloc[train_size:, :], y[train_size:]
        ...
        >>> simple_features = SimpleFeatureEngineer(
        ...     rolling_features=[
        ...         ("wave_height", "mean", 24),
        ...         ("wave_height", "mean", 12),
        ...     ],
        ...     time_features=[
        ...         ("hour_of_day", "cyclical"),
        ...     ],
        ...     keep_original=False,
        ... )
        ...
        >>> model = MLPTimeseriesRegressor(
        ...     predict_ahead=(0,),
        ...     feature_engineer=simple_features,
        ...     verbose=0,
        ... )
        ...
        >>> model.fit(X_train, y_train)  # doctest: +ELLIPSIS
        <keras.src.callbacks.history.History ...
        >>> score_decreases = model.quantile_feature_importances(
        ...     X_test[:100], y_test[:100], n_iter=3, random_state=42)
        >>> # The score decreases of each feature in each iteration
        >>> feature_importances = score_decreases.mean()
        >>> # This will show a barplot of all the score importances, with error bars
        >>> seaborn.barplot(data=score_decreases)  # doctest: +SKIP
        """
        # Model must be fitted for this method
        check_is_fitted(self, "model_")

        if len(self.predict_ahead) > 1:
            raise NotImplementedError(
                "This method is currently not implemented " "for multiple targets"
            )

        if score is None:

            def score(X, y, model=self.model_):
                if self.average_type == "median":
                    return np.mean(np.abs(y - model.predict(X, verbose=0)[:, -1]))
                elif self.average_type == "mean":
                    return np.mean((y - model.predict(X, verbose=0)[:, -1]) ** 2)

        X_transformed = self.preprocess_predict(X, y)

        if self.predict_ahead != [0]:
            y_target = make_shifted_target(y, self.use_diff_of_y, self.predict_ahead).iloc[:, 0]
        else:
            y_target = y.copy().astype(float)

        # Remove rows with missings in either of the two arrays
        missings = np.isnan(y_target) | np.isnan(X_transformed).any(axis=1)
        X_transformed = X_transformed.loc[~missings, :]
        y_target = y_target[~missings]

        base_scores, score_decreases = get_score_importances(
            score,
            X_transformed.to_numpy(),
            y_target.to_numpy(),
            n_iter=n_iter,
            random_state=random_state,
        )

        decreases_df = pd.DataFrame(score_decreases, columns=self.get_feature_names_out())

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
        the created features from `self.get_feature_names_out()`, not the input features.
        To help with this, the explainer comes with a `test_values()` attribute
        that calculates the test values corresponding to the shap values

        Parameters
        ----------
        X: pd.DataFrame
            The dataframe used to 'train' the explainer
        y: pd.Series, optional (default=None)
            Target data used to 'train' the explainer. Only required when self.predict_ahead > 0.
        sample_n: integer, optional (default=None)
            The number of samples to give to the explainer. It is recommended that
            if your background set is greater than 5000, to sample for performance reasons.

        Returns
        -------
        SamShapExplainer:
            Custom Sam object that inherits from shap.DeepExplainer

        Examples
        --------
        >>> import pandas as pd
        >>> import shap
        >>> from sam.models import MLPTimeseriesRegressor
        >>> from sam.feature_engineering import SimpleFeatureEngineer
        >>> from sam.datasets import load_rainbow_beach
        ...
        >>> data = load_rainbow_beach()
        >>> X, y = data, data["water_temperature"]
        >>> test_size = int(X.shape[0] * 0.33)
        >>> train_size = X.shape[0] - test_size
        >>> X_train, y_train = X.iloc[:train_size, :], y[:train_size]
        >>> X_test, y_test = X.iloc[train_size:, :], y[train_size:]
        ...
        >>> simple_features = SimpleFeatureEngineer(
        ...     rolling_features=[
        ...         ("wave_height", "mean", 24),
        ...         ("wave_height", "mean", 12),
        ...     ],
        ...     time_features=[
        ...         ("hour_of_day", "cyclical"),
        ...     ],
        ...     keep_original=False,
        ... )
        ...
        >>> model = MLPTimeseriesRegressor(
        ...     predict_ahead=(0,),
        ...     feature_engineer=simple_features,
        ...     verbose=0,
        ... )
        ...
        >>> model.fit(X_train, y_train)  # doctest: +ELLIPSIS
        <keras.src.callbacks.history.History ...
        >>> ();explainer = model.get_explainer(X_test, y_test, sample_n=10);()
        ... # doctest: +ELLIPSIS
        (...)
        >>> ();shap_values = explainer.shap_values(X_test[0:30], y_test[0:30]);()
        ... # doctest: +ELLIPSIS
        (...)
        >>> test_values = explainer.test_values(X_test[0:30], y_test[0:30])
        >>> shap.plots.force(base_value=float(explainer.expected_value[0]),
        ...                  features=test_values.iloc[-1, :],
        ...                  shap_values=shap_values[-1, :, 0], matplotlib=True)
        """
        import shap

        X_transformed = self.preprocess_predict(X, y, dropna=True)
        if sample_n is not None:
            # Sample some rows to increase performance later
            sampled = np.random.choice(X_transformed.shape[0], sample_n, replace=False)
            X_transformed = X_transformed.iloc[sampled, :]
        explainer = shap.KernelExplainer(self.model_.predict, X_transformed.to_numpy())
        return SamShapExplainer(explainer, self, preprocess_predict=self.preprocess_predict)
