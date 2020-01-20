from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sam.feature_engineering import BuildRollingFeatures, decompose_datetime
from sam.metrics import keras_joint_mse_tilted_loss
from sam.models import create_keras_quantile_mlp
from sam.preprocessing import make_differenced_target, inverse_differenced_target
from sam.utils import FunctionTransformerWithNames

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


class SamShapExplainer(object):

    def __init__(self, explainer, model):
        """
        An object that imitates a SHAP explainer object. (Sort of) implements the base Explainer
        interface `which can be found here
        <https://github.com/slundberg/shap/blob/master/shap/explainers/explainer.py>`_ .
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
        class SamProxyModel():
            fit = None
            use_y_as_feature = model.use_y_as_feature
            feature_names_ = model.get_feature_names()
            preprocess_before_predict = SamQuantileMLP.preprocess_before_predict

        self.model = SamProxyModel()
        # Trick sklearn into thinking this is a fitted variable
        self.model.feature_engineer_ = model.feature_engineer_
        # Will likely be somewhere around 0
        self.expected_value = explainer.expected_value

    def shap_values(self, X, y=None, *args, **kwargs):
        """
        Imitates explainer.shap_values, but combined with the preprocessing from the model.
        Returns a similar format as a regular shap explainer: a list of numpy arrays, one
        for each output of the model.
        """
        X_transformed = self.model.preprocess_before_predict(X, y, dropna=False)
        return self.explainer.shap_values(X_transformed, *args, **kwargs)

    def attributions(self, X, y=None, *args, **kwargs):
        """
        Imitates explainer.attributions, which by default just mirrors shap_values
        """
        return self.shap_values(X, y, *args, **kwargs)

    def test_values(self, X, y=None):
        """
        Only the preprocessing from the model, without the shap values.
        Returns a pandas dataframe with the actual values used for the explaining
        Can be used to better interpret the numpy array that is outputted by shap_values.
        For example, if shap_values outputs a 5x20 numpy array, that means you explained 5 objects
        with 20 features. This function will then return a 5x20 pandas dataframe.
        """
        X_transformed = self.model.preprocess_before_predict(X, y, dropna=False)
        return pd.DataFrame(X_transformed, columns=self.model.feature_names_, index=X.index)


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
    verbose: boolean, optional (default=1)
        The verbosity of fitting the keras model

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

    def get_feature_engineer(self):
        """
        Function that returns an sklearn transformer.

        This is a default, simple columntransformer that:
        - On the time col (if it was passed), does decompose_datetime and cyclicals
        - On the other columns, calculates lag/max/min/mean features for a given window size
        - On the other (nontime) columns, passes them through unchanged (same as lag 0)
        """

        def time_transformer(dates):
            return decompose_datetime(dates, self.timecol,
                                      components=self.time_components,
                                      cyclicals=self.time_cyclicals,
                                      onehots=self.time_onehots,
                                      keep_original=False)

        def identity(x):
            return x

        feature_engineer_steps = \
            [
                # Rolling features
                (rol, BuildRollingFeatures(rolling_type=rol, window_size=self.rolling_window_size,
                                           lookback=0, keep_original=False),
                 self.rolling_cols_) for rol in self.rolling_features
            ] + \
            [
                # Other features
                ("passthrough", FunctionTransformerWithNames(identity, validate=False),
                 self.rolling_cols_)
            ]
        if self.timecol:
            feature_engineer_steps += \
                [("timefeats", FunctionTransformerWithNames(time_transformer, validate=False),
                  [self.timecol])]

        # Drop the time column if exists
        engineer = ColumnTransformer(feature_engineer_steps, remainder='drop')
        # A very simple imputer, for example if a rolling window failed because one of the
        # elements was missing
        imputer = SimpleImputer()
        # Scaling is needed since it vastly improves MLP performance
        scaler = StandardScaler()

        return Pipeline([('columns', engineer), ('impute', imputer), ('scaler', scaler)])

    def get_untrained_model(self):
        """
        A function that returns a simple, 2d keras model.
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
            dropout=self.dropout
        )

    def __init__(self,
                 predict_ahead=1,
                 quantiles=(),
                 use_y_as_feature=True,
                 timecol=None,
                 y_scaler=None,
                 time_components=['minute', 'hour', 'day', 'weekday'],
                 time_cyclicals=['minute', 'hour', 'day'],
                 time_onehots=['weekday'],
                 rolling_window_size=(12,),
                 rolling_features=['mean'],
                 n_neurons=200,
                 n_layers=2,
                 batch_size=16,
                 epochs=20,
                 lr=0.001,
                 dropout=None,
                 momentum=None,
                 verbose=True
                 ):
        self.predict_ahead = predict_ahead
        self.quantiles = quantiles
        self.use_y_as_feature = use_y_as_feature
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

    def validate_data(self, X):
        """
        Void function that validates the data and raises an exception if anything is wrong
        """
        if self.timecol is None:
            warnings.warn(("No timecolumn given. Make sure the data is"
                           "monospaced when given to this model!"), UserWarning)
        else:
            monospaced = X[self.timecol].diff()[1:].unique().size == 1
            if not monospaced:
                raise ValueError("Data is not monospaced, which is required for"
                                 "this model. fit/predict is not possible")

        enough_data = \
            len(self.rolling_window_size) == 0 or X.shape[0] > max(self.rolling_window_size)
        if not enough_data:
            warnings.warn("Not enough data given to caluclate rolling features. "
                          "Output will be entirely missing values.", UserWarning)

    def fit(self, X, y, validation_data=None, **fit_kwargs):
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
        """

        if np.isscalar(self.predict_ahead):
            self.predict_ahead = [self.predict_ahead]

        if not all([p >= 0 for p in self.predict_ahead]):
            raise ValueError("All values of predict_ahead must be 0 or larger!")

        if not y.index.equals(X.index):
            raise ValueError("For training, X and y must have an identical index")

        if not self.use_y_as_feature and self.predict_ahead != [0]:
            raise ValueError("For now, use_y_as_feature must be true unless predict_ahead is 0")

        if self.predict_ahead == [0] and self.use_y_as_feature:
            raise ValueError("use_y_as_feature must be false when predict_ahead is 0")

        self.validate_data(X)

        if self.y_scaler is not None:
            y = pd.Series(self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel(),
                          index=y.index)

        if self.use_y_as_feature:
            X = X.assign(y_=y.copy())

        # Create the actual target
        if (self.use_y_as_feature) and (self.predict_ahead != [0]):
            target = make_differenced_target(y, self.predict_ahead)
        else:
            # Dataframe with 1 column. Will use y's index and name
            target = pd.DataFrame(y.copy())
        # Index where target is nan, cannot be trained on.
        targetnanrows = target.isna().any(axis=1)

        # buildrollingfeatures
        self.rolling_cols_ = [col for col in X if col != self.timecol]
        self.feature_engineer_ = self.get_feature_engineer()

        # Apply feature engineering
        X_transformed = self.feature_engineer_.fit_transform(X)
        # Now we have fitted the feature engineer, we can set the feature names
        self._feature_names = self.set_feature_names(X, X_transformed)

        # Now feature names are set, we can start using self.get_feature_names()
        X_transformed = pd.DataFrame(X_transformed,
                                     columns=self.get_feature_names(),
                                     index=X.index)
        self.n_inputs_ = len(self.get_feature_names())

        # Create output column names. In this model, our outputs are assumed to have the
        # form: [quantile_1_output_1, quantile_1_output_2, ... ,
        # quantile_n_output_1, quantile_n_output_2, ..., mean_output_1, mean_output_2]
        # Where n_output (1 or 2 in this example) is decided by self.predict_ahead
        self.prediction_cols_ = \
            ['predict_lead_{}_q_{}'.format(p, q)
             for q in self.quantiles for p in self.predict_ahead]
        self.prediction_cols_ += ['predict_lead_{}_mean'.format(p) for p in self.predict_ahead]
        self.n_outputs_ = len(self.prediction_cols_)

        # Remove the first n rows because they are nan anyway because of rolling features
        if len(self.rolling_window_size) > 0:
            X_transformed = X_transformed.iloc[max(self.rolling_window_size):]
            target = target.iloc[max(self.rolling_window_size):]
        # Filter rows where the target is unknown
        X_transformed = X_transformed.loc[~targetnanrows]
        target = target.loc[~targetnanrows]

        # TODO imputing the data, Daan knows many methods
        assert X_transformed.isna().sum().sum() == 0, \
            "Data cannot contain nans. Imputation not supported for now"

        self.model_ = self.get_untrained_model()

        # Create validation data:
        if validation_data is not None:
            X_val, y_val = validation_data

            if self.y_scaler is not None:
                y_val = pd.Series(self.y_scaler.transform(y_val.values.reshape(-1, 1)).ravel(),
                                  index=y_val.index)

            # This does not affect training: it only calls transform, not fit
            X_val_transformed = self.preprocess_before_predict(X_val, y_val)
            X_val_transformed = pd.DataFrame(X_val_transformed,
                                             columns=self.get_feature_names(), index=X_val.index)
            if (self.use_y_as_feature) and (self.predict_ahead != [0]):
                y_val_transformed = make_differenced_target(y_val, self.predict_ahead)
            else:
                # Dataframe with 1 column. Will use y's index and name
                y_val_transformed = pd.DataFrame(y_val.copy())

            # The lines below are only to deal with nans in the validation set
            # These should eventually be replaced by Arjans/Fennos function for removing nan rows
            # So that this code will be much more readable
            targetnanrows = y_val_transformed.isna().any(axis=1)
            # Remove the first n rows because they are nan anyway because of rolling features
            if len(self.rolling_window_size) > 0:
                X_val_transformed = X_val_transformed.iloc[max(self.rolling_window_size):]
                y_val_transformed = y_val_transformed.iloc[max(self.rolling_window_size):]
            # Filter rows where the target is unknown
            X_val_transformed = X_val_transformed.loc[~targetnanrows]
            y_val_transformed = y_val_transformed.loc[~targetnanrows]
            # Until here
            validation_data = (X_val_transformed, y_val_transformed)

        # Fit model
        history = self.model_.fit(X_transformed, target, batch_size=self.batch_size,
                                  epochs=self.epochs, verbose=self.verbose,
                                  validation_data=validation_data, **fit_kwargs)
        return history

    def preprocess_before_predict(self, X, y, dropna=False):
        """
        Create a dataframe that can be fed to self.model_. This is useful for several usecases,
        where you want to use the underlying keras model as opposed to the wrapper.
        For example shap, eli5, and even just implementing the `predict` function.
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, 'feature_engineer_')

        assert y is None or y.index.equals(X.index), \
            "For predicting, X and y must have an identical index"
        if self.use_y_as_feature:
            X = X.assign(y_=y.copy())
        X_transformed = self.feature_engineer_.transform(X)
        if dropna:
            X_transformed = X_transformed[~np.isnan(X_transformed).any(axis=1)]
        return X_transformed

    def predict(self, X, y=None):
        """
        Make a prediction, and optionally undo differencing
        Important! This is different from sklearn/tensorflow API...
        We need y during prediction for two reasons:
        1) a lagged version is used for feature engineering
        2) The underlying model predicts a differenced number, and we want to output the 'real'
           prediction, so we need y to undo the differencing
        Keep in mind that prediction will work if you are predicting the future. e.g. you have
        data from 00:00-12:00, and are predicting 4 hours into the future, it will predict what
        the value will be at 4:00-16:00
        """
        assert self.predict_ahead == 0 or y is not None, \
            "When predict_ahead > 0, y is needed for prediction"

        assert y is None or X.index.equals(y.index), \
            "For predicting, X and y must have an identical index"

        self.validate_data(X)

        X_transformed = self.preprocess_before_predict(X, y)
        prediction = self.model_.predict(X_transformed)

        # Put the predictions in a dataframe so we can undo the differencing
        prediction = pd.DataFrame(prediction,
                                  columns=self.prediction_cols_,
                                  index=X.index)
        # This parameter decides if we did differencing or not, so undo it if we used it
        if self.use_y_as_feature:
            prediction = inverse_differenced_target(prediction, y)

        if prediction.shape[1] == 1:
            # If you just wanted to predict a single value without quantiles,
            # just return a series. Easier to work with.
            prediction = prediction.iloc[:, 0]

            if self.y_scaler is not None:
                prediction = pd.Series(self.y_scaler.inverse_transform(
                    prediction.values).ravel(), index=prediction.index)

        else:
            if self.y_scaler is not None:
                inv_pred = np.zeros_like(prediction)
                for i in range(prediction.shape[1]):
                    inv_pred[:, i] = self.y_scaler.inverse_transform(
                        prediction.iloc[:, i].values.reshape(-1, 1)).ravel()
                prediction = pd.DataFrame(
                    inv_pred, columns=prediction.columns, index=prediction.index)

        return prediction

    def set_feature_names(self, X, X_transformed):
        """
        Default function for setting the feature names

        For the default feature engineer, it outputs features like 'mean__Q#mean_1'
        Which is much easier to interpret if we remove the 'mean__'
        """
        names = self.feature_engineer_.named_steps['columns'].get_feature_names()
        names = [colname.split('__')[1] for colname in names]
        return names

    def get_feature_names(self):
        """
        Function for obtaining feature names. This can be used instead of the
        attribute. More widely used than an attribute, and more
        compatible with the sklearn API.
        """
        check_is_fitted(self, '_feature_names')
        return self._feature_names

    def get_actual(self, y):
        """
        Convenience function for getting an actual value. Mainly useful for scoring the model
        This essentially does and undoes differencing on y, meaning this function will output what
        a perfect model would have outputted.
        If predict_ahead is 0, no differencing is done anyway, so y is just returned unchanged.

        If self.predict_ahead is a single value, this function will return a series.
        If self.predict_ahead is multiple values, this function will return a dataframe.
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, 'model_')
        if len(self.predict_ahead) == 1:
            pred = self.predict_ahead[0]
        else:
            pred = self.predict_ahead

        if self.use_y_as_feature:
            target = make_differenced_target(y, pred)
            actual = inverse_differenced_target(target, y)
        else:
            actual = y.copy()

        return actual

    def score(self, X, y):
        """
        Default score function. Use sum of mse and tilted loss
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, 'model_')
        # We need a dataframe, regardless of if these functions outputs a series or dataframe
        prediction = pd.DataFrame(self.predict(X, y))
        actual = pd.DataFrame(self.get_actual(y))

        # actual usually has some missings at the end
        # prediction usually has some missings at the beginning
        # We ignore the rows with missings
        missings = actual.isna().any(axis=1) | prediction.isna().any(axis=1)
        actual = actual.loc[~missings]
        prediction = prediction.loc[~missings]

        # self.prediction_cols_[-1] defines the mean prediction
        # For n outputs, we need the last n columns instead
        # Therefore, this line calculates the mse of the mean prediction
        mean_prediction = prediction[self.prediction_cols_[-1*len(self.predict_ahead):]].values
        # Calculate the MSE of all the predictions, and tilted loss of quantile predictions,
        # then sum them at the end
        loss = np.sum(np.mean((actual - mean_prediction)**2, axis=0))
        for i, q in enumerate(self.quantiles):
            startcol = len(self.predict_ahead)*i
            endcol = startcol+len(self.predict_ahead)
            e = np.array(actual - prediction[self.prediction_cols_[startcol:endcol]].values)
            # Calculate all the quantile losses, and sum them at the end
            loss += np.sum(np.mean(np.max([q*e, (q-1)*e], axis=0), axis=0))
        return loss

    def dump(self, foldername, prefix='model'):
        """
        Writes the following files:
        * prefix.pkl
        * prefix.h5

        to the folder given by foldername. prefix is configurable, and is
        'model' by default
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, 'model_')

        import cloudpickle

        foldername = Path(foldername)

        # TEMPORARY
        self.model_.save(foldername / (prefix + '.h5'))

        # Set the models to None temporarily, because they can't be pickled
        backup, self.model_ = self.model_, None

        with open(foldername / (prefix + '.pkl'), 'wb') as f:
            cloudpickle.dump(self, f)

        # Set it back
        self.model_ = backup

    def _get_loss(self):
        """
        Convenience function, mirrors create_keras_quantile_mlp
        Only needed for loading, since it is a custom object, it is not
        saved in the .h5 file by default
        """
        if len(self.quantiles) == 0:
            mse_tilted = 'mse'
        else:
            def mse_tilted(y, f):
                loss = keras_joint_mse_tilted_loss(y, f, self.quantiles, len(self.predict_ahead))
                return(loss)
        return mse_tilted

    @classmethod
    def load(cls, foldername, prefix='model'):
        """
        Reads the following files:
        * prefix.pkl
        * prefix.h5

        from the folder given by foldername. prefix is configurable, and is
        'model' by default
        Output is an entire instance of the fitted model that was saved
        """
        import cloudpickle
        from tensorflow.keras.models import load_model

        foldername = Path(foldername)
        with open(foldername / (prefix + '.pkl'), 'rb') as f:
            obj = cloudpickle.load(f)

        loss = obj._get_loss()
        obj.model_ = load_model(foldername / (prefix + '.h5'),
                                custom_objects={'mse_tilted': loss})
        return obj

    def summary(self, print_fn=print):
        """
        Combines several methods of summary to create a 'wrapper' summary method.
        """
        # This function only works if the estimator is fitted
        check_is_fitted(self, 'model_')
        print_fn(str(self))
        print_fn(self.get_feature_names())
        self.model_.summary(print_fn=print_fn)

    def quantile_feature_importances(self, X, y, score=None, n_iter=5):
        """
        Computes feature importances based on the quantile loss function.
        This function uses `ELI5's 'get_score_importances (link)
        <https://eli5.readthedocs.io/en/latest/autodocs/permutation_importance.html>`_
        to compute feature importances. It is a method that measures how the score decreases when
        a feature is not available.
        This is essentially a model-agnostic type of feature importance that works with
        every model, including keras MLP models.

        Parameters
        ----------
        X: pd.DataFrame
              dataframe with test or train features
        y: pd.Series
              dataframe with test or train target
        score: function, optional (default=None)
             function with signature score(X, y)
             that returns a scalar. Will be used to measure score decreases for ELI5.
             By default, use the same scoring as is used by self.score(): RMSE plus MAE of
             quantiles.
        n_iter: int, optional (default=5)
             Number of iterations to use for ELI5. Since ELI5 results can vary wildly, increasing
             this parameter may provide more stablitity at the cost of a longer runtime

        Returns
        -------
        score_decreases: Pandas dataframe,  shape (n_iter x n_features)
            The score decreases when leaving out each feature per iteration. The higher, the more
            important each feature is considered by the model.

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
        check_is_fitted(self, 'model_')
        if len(self.predict_ahead) > 1:
            raise NotImplementedError("This method is currently not implemented "
                                      "for multiple targets")

        from eli5.permutation_importance import get_score_importances

        if score is None:
            # By default, use RMSE + quantile MAE
            def score(X, y, model=self.model_):
                # quantile loss function that performs on the transformed model/data rather than a
                # 'wrapper'. X is the transformed data, y is the target, model is a keras model,
                # qs are the quantiles
                y_pred = model.predict(X)
                loss = np.sqrt(np.mean((y - y_pred[:, -1])**2))
                for i, q in enumerate(self.quantiles):
                    e = np.array(y - y_pred[:, i])
                    loss += np.mean(np.max([q*e, (q-1)*e], axis=0))
                return loss

        X_transformed = self.preprocess_before_predict(X, y)
        if self.use_y_as_feature:
            y_target = make_differenced_target(y, self.predict_ahead).iloc[:, 0]
        else:
            y_target = y.copy()

        # Remove rows with missings in either of the two arrays
        missings = np.isnan(y_target) | np.isnan(X_transformed).any(axis=1)
        X_transformed = X_transformed[~missings, :]
        y_target = y_target[~missings]

        # use eli5 to compute feature importances:
        _, score_decreases = get_score_importances(score, X_transformed, y_target, n_iter=n_iter)

        decreases_df = pd.DataFrame(score_decreases, columns=self.get_feature_names())

        return decreases_df

    def get_explainer(self, X, y=None, sample_n=None):
        """
        Obtain a shap explainer-like object. This object can be used to
        create shap values and explain predictions.

        Keep in mind that this will explain
        the created features from `self.get_feature_names()`, not the input features.
        To help with this, the explainer comes with a `test_values()` attribute
        that calculates the test values corresponding to the shap values

        Parameters
        ----------
        X: array-like
            The dataframe used to 'train' the explainer
        y: array-like, optional (default=None)
            Only required when self.predict_ahead > 0. Used to 'train' the explainer.
        sample_n: integer, optional (default=None)
            The number of samples to give to the explainer. It is reccommended that
            if your background set is greater than 5000, to sample for performance reasons.

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
