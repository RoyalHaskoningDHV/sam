from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer

from sam.feature_engineering import BuildRollingFeatures, decompose_datetime
from sam.metrics import keras_joint_mse_tilted_loss
from sam.models import create_keras_quantile_mlp
from sam.preprocessing import make_differenced_target, inverse_differenced_target

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


class SamQuantileMLP(BaseEstimator):
    """
    This is an example class for how the SAM skeleton can work. This is not the final/only model,
    there are some notes:
    - First, this model only predicts a single target (plus quantiles). For example, we can
    predict 4 timesteps into the future, with mean/5%/95% percentiles. But not 4 and 5 timesteps.
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
    predict_ahead: integer, optional (default=1)
        how many steps to predict ahead. If 0, predict the present, without differencing. If >0,
        predict the future, with differencing.
    quantiles: array-like, optional (default=())
        The quantiles to predict. Between 0 and 1. Keep in mind that the mean will be predicted
        regardless of this parameter
    use_y_as_feature: boolean, optional (default=True)
        Whether or not to use y as a feature for predicting. If predict_ahead=0, this must be False
    timeecol: string, optional (default=None)
        If not None, the column to use for constructing time features. For now,
        creating features from a DateTimeIndex is not supported yet.
    time_components: array-like, optional (default=('minute', 'hour', 'month', 'weekday'))
        The timefeatures to create. See :ref:`decompose_datetime`.
    time_cyclicals: array-like, optional (default=('minute', 'hour', 'month', 'weekday'))
        The cyclical timefeatures to create. See :ref:`decompose_datetime`.
    rolling_window_size: array-like, optional (default=(5,))
        The window size to use for `BuildRollingFeatures`
    n_neurons: integer, optional (default=200)
        The number of neurons to use in the model, see :ref:`create_keras_quantile_mlp`
    n_layers: integer, optional (default=2)
        The number of layers to use in the model, see :ref:`create_keras_quantile_mlp`
    batch_size: integer, optional (default=16)
        The batch size to use in the model, see :ref:`create_keras_quantile_mlp`
    epochs: integer, optional (default=20)
        The number of epochs to use in the model, see :ref:`create_keras_quantile_mlp`
    lr: integer, optional (default=0.001)
        The learning rate to use in the model, see :ref:`create_keras_quantile_mlp`
    dropout: integer, optional (default=None)
        The type of dropout to use in the model, see :ref:`create_keras_quantile_mlp`
    momentum: integer, optional (default=None)
        The type of momentum in the model, see :ref:`create_keras_quantile_mlp`
    verbose: boolean, optional (default=1)
        The verbosity of fitting the keras model

    Attributes
    ----------
    feature_engineer_: Sklearn transformer
        The transformer used on the raw data before prediction
    n_inputs_: integer
        The number of inputs used for the underlying neural network
    n_outputs: integer
        The number of outputs (columns) from the model
    prediction_cols: array of strings
        The names of the output columns from the model.
    model_: Keras model
        The underlying keras model

    Examples
    --------
    >>> from sam.models import SamQuantileMLP
    >>> from sam.data_sources import read_knmi
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error

    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', latitude=52.11, longitude=5.18, freq='hourly',
    >>>                 variables=['FH', 'FF', 'FX', 'T', 'TD', 'SQ', 'Q', 'DR', 'RH', 'P',
    >>>                            'VV', 'N', 'U', 'IX', 'M', 'R', 'S', 'O', 'Y'])
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

    >>> # We are predicting the weather 1 hour ahead. Since weather is highly autocorrelated, we
    >>> # expect this the persistence benchmark to score decently high, but also it should be
    >>> # easy to predict the weather 1 hour ahead, so the model should do even better.
    >>> model = SamQuantileMLP(predict_ahead=1, use_y_as_feature=True, timecol='TIME',
    >>>                       quantiles=[0.25, 0.75], epochs=5,
    >>>                       time_components=['hour', 'month', 'weekday'],
    >>>                       time_cyclicals=['hour', 'month', 'weekday'],
    >>>                       rolling_window_size=[1,5,6])

    >>> # fit returns a keras history callback object, which can be used as normally
    >>> history = model.fit(X_train, y_train)
    >>> pred = model.predict(X_test, y_test).dropna()

    >>> actual = model.get_actual(y_test).dropna()
    >>> # Because of impossible to know values, some rows have to be dropped. After dropping
    >>> # them, make sure the indexes still match by dropping the same rows from each side
    >>> pred, actual = pred.reindex(actual.index).dropna(), actual.reindex(pred.index).dropna()
    >>> mean_squared_error(actual, pred.iloc[:, -1])  # last column contains mean prediction
    114.50628975834859

    >>> # Persistence corresponds to predicting the present, so use ytest
    >>> persistence_prediction = y_test.reindex(actual.index).dropna()
    >>> mean_squared_error(actual, persistence_prediction)
    149.35018919848642

    >>> # As we can see, the model performs significantly better than the persistence benchmark
    >>> # Mean benchmark, which does much worse:
    >>> mean_prediction = pd.Series(y_test.mean(), index = actual)
    >>> mean_squared_error(actual, mean_prediction)
    2410.20138157309
    """

    class FunctionTransformerWithNames(FunctionTransformer):
        """
        Utility class. Used in the default Feature Engineer.
        Acts just like FunctionTransformer, but with `get_feature_names`

        The feature engineer should have a `get_feature_names()` attribute. ColumnTransformers
        have this by default, but only if all the substeps also have the attribute.
        BuildRollingFeatures already has it, but the FunctionTransformer doesn't have it yet, so
        we add it here.
        """
        def transform(self, X, y='deprecated'):
            output = super().transform(X)
            self._feature_names = list(output.columns.values)
            return output

        def get_feature_names(self):
            return self._feature_names

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
                                      cyclicals=self.time_cyclicals, keep_original=False)

        def identity(x):
            return x

        feature_engineer_steps = [
            # Lag features
            ("lag", BuildRollingFeatures(rolling_type='lag', window_size=self.rolling_window_size,
                                         lookback=0, keep_original=False),
             self.rolling_cols_),
            ("max", BuildRollingFeatures(rolling_type='max', window_size=self.rolling_window_size,
                                         keep_original=False),
             self.rolling_cols_),
            ("min", BuildRollingFeatures(rolling_type='min', window_size=self.rolling_window_size,
                                         keep_original=False),
             self.rolling_cols_),
            ("mean", BuildRollingFeatures(rolling_type='mean',
                                          window_size=self.rolling_window_size,
                                          keep_original=False),
             self.rolling_cols_),

            # Other features
            ("passthrough", self.FunctionTransformerWithNames(identity, validate=False),
             self.rolling_cols_)
        ]
        if self.timecol:
            feature_engineer_steps += \
                [("timefeats", self.FunctionTransformerWithNames(time_transformer, validate=False),
                  [self.timecol])]

        # Drop the time column if exists
        return ColumnTransformer(feature_engineer_steps, remainder='drop')

    def get_untrained_model(self):
        """
        A function that returns a simple, 2d keras model.
        This is just a wrapper for sam.models.create_keras_quantile_mlp
        """
        return create_keras_quantile_mlp(
            n_input=self.n_inputs_,
            n_neurons=self.n_neurons,
            n_layers=self.n_layers,
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
                 time_components=('minute', 'hour', 'month', 'weekday'),
                 time_cyclicals=('minute', 'hour', 'month', 'weekday'),
                 rolling_window_size=(5,),
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
        self.time_components = time_components
        self.time_cyclicals = time_cyclicals
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

    def fit(self, X, y):
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
        """

        if not np.isscalar(self.predict_ahead):
            raise ValueError("For now, multiple timestep predictions are not supported")

        if not y.index.equals(X.index):
            raise ValueError("For training, X and y must have an identical index")

        if self.use_y_as_feature and self.predict_ahead == 0:
            raise ValueError("For now, when predict_ahead=0, you cannot also use y as a feature")

        self.validate_data(X)

        if self.use_y_as_feature:
            X = X.assign(y_=y.copy())

        # Create the actual target
        if self.predict_ahead > 0:
            target = make_differenced_target(y, self.predict_ahead)
        else:
            target = y.copy()
        # Index where target is nan, cannot be trained on.
        targetnanrows = target.isna()

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

        # Create output column names, depends on sam.models.create_keras_quantile_mlp
        self.prediction_cols_ = \
            ['predict_lag_{}_q_{}'.format(self.predict_ahead, q) for q in self.quantiles]
        self.prediction_cols_ += ['predict_lag_{}_mean'.format(self.predict_ahead)]
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

        # Fit model
        self.model_.fit(X_transformed, target, batch_size=self.batch_size,
                        epochs=self.epochs, verbose=self.verbose)
        return self

    def preprocess_before_predict(self, X, y, dropna=False):
        """
        Create a dataframe that can be fed to self.model_. This is useful for several usecases,
        where you want to use the underlying keras model as opposed to the wrapper.
        For example shap, eli5, and even just implementing the `predict` function.
        """

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
        if self.predict_ahead > 0:
            prediction = inverse_differenced_target(prediction, y)
        else:
            prediction = prediction.copy()

        # We either return a series or a dataframe depending on if there are quantiles
        if self.quantiles == []:
            # Only one output, return 1d numpy array
            return prediction['predict_lag_{}_mean'.format(self.predict_ahead)]
        else:
            return prediction

    def set_feature_names(self, X, X_transformed):
        """
        Default function for setting the feature names
        """
        return self.feature_engineer_.get_feature_names()

    def get_feature_names(self):
        """
        Function for obtaining feature names. More widely used than an attribute, and more
        compatibly with the sklearn API
        """
        check_is_fitted(self, '_feature_names')
        return self._feature_names

    def get_actual(self, y):
        """
        Convenience function for getting an actual value. Mainly useful for scoring the model
        This essentially does and undoes differencing on y, meaning this function will output what
        a perfect model would have outputted.
        If predict_ahead is 0, no differencing is done anyway, so y is just returned unchanged.
        """
        if self.predict_ahead > 0:
            target = make_differenced_target(y, self.predict_ahead)
            actual = inverse_differenced_target(target, y)
        else:
            actual = y.copy()
        return actual

    def score(self, X, y):
        """
        Default score function. Use sum of rmse and tilted loss
        """
        # quantile loss function
        prediction = self.predict(X, y)
        actual = self.get_actual(y)

        # actual usually has some missings at the end
        # prediction usually has some missings at the beginning
        # We ignore the rows with missings
        missings = actual.isna() | prediction.isna().any(axis=1)
        actual = actual.loc[~missings]
        prediction = prediction.loc[~missings]

        # self.prediction_cols_[-1] defines the mean prediction
        # Therefore, this line calculates the rmse of the mean prediction
        loss = np.sqrt(np.mean((actual - prediction[self.prediction_cols_[-1]])**2))
        for i, q in enumerate(self.quantiles):
            e = np.array(actual - prediction[self.prediction_cols_[i]])
            loss += np.mean(np.max([q*e, (q-1)*e], axis=0))
        return loss

    def dump(self, foldername, prefix='model'):
        """
        Writes the following files:
        * prefix.pkl
        * prefix.h5

        to the folder given by foldername. prefix is configurable, and is
        'model' by default
        """
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
                loss = keras_joint_mse_tilted_loss(y, f, self.quantiles)
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
        print_fn(str(self))
        print_fn(self.get_feature_names())
        self.model_.summary(print_fn=print_fn)
