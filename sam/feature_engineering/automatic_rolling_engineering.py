# sam functionality
from sam.feature_engineering import BuildRollingFeatures
from sam.feature_engineering import decompose_datetime
# sklearn functionality
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression, BayesianRidge
# other
import pandas as pd
import numpy as np


class AutomaticRollingEngineering(BaseEstimator, TransformerMixin):
    """
    Steps for automatic rolling engineering:

    - setup self.n_rollings number of different rolling features (unparameterized yet) in
        sklearn ColumnTransformer pipeline
    - find the best parameters for each of the rolling features using random search
    - setup a ColumnTransformer with these best features that can be used in the transform method

    Parameters
    ----------
    window_sizes: list of lists
        each list should be integers or one of one of scipy.stats.distributions that convert to
        a window_size for BuildRollingFeatures.
        Each sublist corresponds to range tried for the n_rollings, and should be non-overlapping.
        So if you want 2 rollings to be generated per rolling_type and per feature, this could be:
        [scipy.stats.randint(1, 24), scipy.stats.randint(24, 168)].
        Note that using long lists results in overflow error, therefore randint is recommended.
    rolling_types: list of strings (default=['mean', 'lag'])
        rolling_types to try for BuildRollingFeatures.
        Note: cannot be 'ewm'.
    estimator_type: str (default='lin')
        type of estimator to determine rolling importance. Can one of: ['rf', 'lin', 'bayeslin']
    n_iter_per_param: int (default=25)
        number of random values to try for each parameter. The total number of iterations is
        given by n_iter_per_param * len(window_sizes) * len(rolling_types)
    cv: int (default=3)
        number of cross-validated tries to attempt for each parameter combination
    passthrough: bool (default=True)
        whether to pass original features in the transform method or not
    cyclicals: list (default=[])
        A list of pandas datetime properties, such as ['minute', 'hour', 'dayofweek', 'week'],
        that will be converted to cyclicals.
        The rationale here is that if time features are not added, the rolling engineering
        will find values for instance of 1 day ago to predict well, while actually this is simply
        a recurring daily pattern that can be captured by time features.
        Note that if timefeatures are added, they are not added in the transform method. Therefore,
        you will have to add them yourself during subsequent model building stages.
    onehots: list (default=[])
        A list of pandas datetime properties, such as ['minute', 'hour', 'dayofweek', 'week'],
        that will be converted using onehot encoding.
        The rationale here is that if time features are not added, the rolling engineering
        will find values for instance of 1 day ago to predict well, while actually this is simply
        a recurring daily pattern that can be captured by time features.
        Note that if timefeatures are added, they are not added in the transform method. Therefore,
        you will have to add them yourself during subsequent model building stages.

    Attributes
    ----------
    feature_importances_: pandas dataframe
        With 'feature_name' column and
        if estimator_type is set to 'rf', a 'coefficients' column
        if estimator_type is set to 'lin', an 'importances' column
    feature_names_: list of strings
        names of all features, depends on self.passthrough
    rolling_feature_names_: list of strings
        names of rolling features that were added

    Examples
    --------
    >>> from sam.data_sources import read_knmi
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> from sam.feature_engineering import AutomaticRollingEngineering
    >>> from sklearn.model_selection import train_test_split
    >>> from scipy.stats import randint

    >>> # load some data
    >>> data = read_knmi(
    >>>     '2018-01-01',
    >>>     '2019-01-01',
    >>>     variables = ['T', 'FH', 'FF', 'FX', 'SQ', 'Q', 'DR', 'RH']).set_index(['TIME'])

    >>> # let's predict temperature 12 hours into the future
    >>> target = 'T'
    >>> fut = 12
    >>> y = data[target].shift(-fut).iloc[:-fut]
    >>> X = data.iloc[:-fut]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    >>> # do the feature selection, first try without adding timefeatures
    >>> ARE = AutomaticRollingEngineering(
    >>>     window_sizes=[randint(1,24)], rolling_types=['mean', 'lag'])
    >>> ARE.fit(X_train, y_train)

    >>> # check some diagnostics, note: results may vary as it is a random search
    >>> r2_base, r2_rollings, yhat_base, yhat_roll = ARE.compute_diagnostics(
    >>>     X_train, X_test, y_train, y_test)
    >>> print(r2_base, r2_rollings)
    0.3435308160142201 0.7113915967302382

    >>> # you can also inspect feature importances:
    >>> sns.barplot(data=ARE.feature_importances_, y='feature_name', x='coefficients')

    >>> # and make plot of the timeseries:
    >>> plt.figure(figsize=(12, 6))
    >>> plt.plot(X_test.index, y_test.ravel(), 'ok', label='data')
    >>> plt.plot(X_test.index, yhat_base, lw=3, alpha=0.75, label='yhat_base (r2: %.2f)'%r2_base)
    >>> plt.plot(
    >>>     X_test.index, yhat_roll, lw=3, alpha=0.75, label='yhat_rolling (r2: %.2f)'%r2_rollings)
    >>> plt.legend(loc='best')
    """

    def __init__(self,
                 window_sizes,
                 rolling_types=['mean', 'lag'],
                 n_iter_per_param=25,
                 cv=3,
                 estimator_type='lin',
                 passthrough=True,
                 cyclicals=[],
                 onehots=[]):

        self.window_sizes = window_sizes
        self.rolling_types = rolling_types
        self.n_iter_per_param = n_iter_per_param
        self.cv = cv
        self.estimator_type = estimator_type
        self.passthrough = passthrough
        self.add_time_features = cyclicals + onehots
        self.cyclicals = cyclicals
        self.onehots = onehots

    def _setup_estimator(self):
        if self.estimator_type == 'rf':
            estimator = RandomForestRegressor(n_estimators=100, min_samples_split=5)
        elif self.estimator_type == 'lin':
            estimator = LinearRegression()
        elif self.estimator_type == 'bayeslin':
            estimator = BayesianRidge()
        return estimator

    def _setup_pipeline(self, original_features, time_features):
        """
        Create the pipeline that is later fitted. Includes:
        - BuildRollingFeatures
        - SimpleImputer
        - Model (depending on self.estimator_type)
        """
        rolls = []
        for feature in original_features:
            for i in range(len(self.window_sizes)):
                # only keep originals once to prevent doubling
                rolls.append(('%s_%d' % (feature, i),
                              BuildRollingFeatures(keep_original=(i == 0)), [feature]))

        class UnitTransformer(BaseEstimator, TransformerMixin):
            """
            This transformer does nothing but pass on the features
            This is required to pass on the time features without changing them
            (the Columntransformer passthrough option does not preserve feature names)
            """

            def __init__(self):
                pass

            def get_feature_names(self):
                check_is_fitted(self, 'feature_names_')
                return self.feature_names_

            def fit(self, X, y):
                self.feature_names_ = X.columns
                return self

            def transform(self, X):
                return X

        for time_feature in time_features:
            rolls.append((time_feature, UnitTransformer(), [time_feature]))

        pipeline = Pipeline(steps=[
            ('rollpipe', ColumnTransformer(rolls, n_jobs=-1)),
            ('imputer', SimpleImputer()),
            ('model', self._setup_estimator())])

        return pipeline

    def _setup_rolling_gridsearch_params(self, original_features):
        """
        create grid to search over
        """
        param_grid = {}
        for i in range(len(self.window_sizes)):
            for feature in original_features:
                param_grid['rollpipe__%s_%d__window_size' % (feature, i)] = self.window_sizes[i]
                param_grid['rollpipe__%s_%d__rolling_type' % (feature, i)] = self.rolling_types

        return param_grid

    def get_feature_names(self):
        check_is_fitted(self, 'feature_names_')
        return self.feature_names_

    def _validate_params(self, X_shape):

        for window_size in self.window_sizes:

            if type(window_size) is list:
                interval = [np.min(window_size), np.max(window_size)]
            else:
                # otherwise we assume it is a scipy.stats distribution
                interval = window_size.interval(1)

            assert interval[0] >= 0,\
                'window_size must be greater than 0'

            assert X_shape[0] > interval[1],\
                'number of samples too small for maximum window_size ' +\
                '(should be smaller than X.shape[0])'

        assert 'ewm' not in self.rolling_types, 'rolling_type cannot be ewm'
        assert self.n_iter_per_param > 0, 'n_iter_per_param must be greater than 0'
        assert self.cv > 1, 'cv must be greater than 1'

    def _add_time_features(self, X):
        """
        This function adds the timefeatures given in 'components' to X.

        Parameters
        ----------
        X: pandas dataframe
            with time index and other features as columns
        components: list of strings
            should be pandas dt properties

        Returns
        -------
        X: pandas dataframe
            original X dataframe with the time features added
        """

        if len(self.add_time_features) > 0:
            times_df = pd.DataFrame({'TIME': X.index})
            time_features = decompose_datetime(
                times_df, components=self.add_time_features,
                onehots=self.onehots, cyclicals=self.cyclicals, keep_original=True)
            time_features = time_features.set_index('TIME', drop=True)

            # some columns can be constant (i.e. higher res than data is in). However, removing
            # all constant columns will also remove one-hot-encoded features that are not
            # currently in the dataset (like week_52 if you only have partial year data).
            # So, we simply leave all constant features in there.

            X = X.join(time_features)
            timecols = time_features.columns
        else:
            timecols = []

        return X, timecols

    def fit(self, X, y):
        """
        Finds the best rolling feature parameters and sets up transformer for the transform method
        Note!: all input must be linearly increasing in time and have a datetime index.

        Parameters
        ----------
        X: pandas dataframe
            with shape [n_samples x n_features]
        y: pandas dataframe
            with shape [n_samples]
        """

        # save feature names here before adding timefeatures
        original_features = X.columns

        # add time time features
        X, timefeatures = self._add_time_features(X)

        self._validate_params(np.shape(X))

        # setup pipeline
        pipeline = self._setup_pipeline(original_features, timefeatures)
        param_grid = self._setup_rolling_gridsearch_params(original_features)

        n_iter = self.n_iter_per_param * len(self.window_sizes) * len(self.rolling_types)

        # find best rollings using randomsearch (future improvement: use BayesSearchCV from skopt)
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_jobs=-1,
            cv=TimeSeriesSplit(self.cv), verbose=2, iid=False, n_iter=n_iter)
        search.fit(X, y)

        # recover feature names
        feature_names = search.best_estimator_['rollpipe'].get_feature_names()

        # fix names from e.g. 'RH_1__RH#mean_96' to 'RH#mean_96'
        feature_names = [f.split('__')[1] for f in feature_names]
        self.rolling_feature_names_ = [f for f in feature_names if '#' in f]

        # convert these names to a set of BuildRollingFeatures that creates these features
        transformers = []
        for i, f in enumerate(self.rolling_feature_names_):

            feature_name = f.split('#')[0]
            rolling_type = f.split('#')[1].split('_')[0]
            window_size = int(f.split('_')[-1])

            transformers.append([
                '%s_%s_%d' % (feature_name, rolling_type, window_size),
                BuildRollingFeatures(
                    rolling_type=rolling_type,
                    window_size=window_size,
                    keep_original=False),
                [feature_name]
            ])

        # now save the columntransformer to the self object to be used in the transform method
        self._transformer = ColumnTransformer(transformers, n_jobs=-1)
        self._transformer = self._transformer.fit(X)

        X_new = search.best_estimator_[:-1].transform(X)
        if self.estimator_type == 'rf':
            imp_name = 'importances'
            importances = search.best_estimator_['model'].feature_importances_
        elif self.estimator_type == 'lin':
            imp_name = 'coefficients'
            importances = search.best_estimator_['model'].coef_

        # also save the feature importances under convenient name:
        self.feature_importances_ = pd.DataFrame({
            imp_name: importances,
            'feature_name': feature_names}).sort_values(by=imp_name, ascending=False)

        # this is required for setup of df including timefeatures in transform
        if self.passthrough:
            self.feature_names_full_ = feature_names
        else:
            self.feature_names_full_ = self.rolling_feature_names_

        # these are the final feature names
        self.timecols = [f for f in self.feature_names_full_ if 'TIME' in f]
        self.feature_names_ = [f for f in self.feature_names_full_ if 'TIME' not in f]
        return self

    def transform(self, X):
        """
        Applies the BuildRollingFeature transformations found to work best in the fit method
        Note!: all input must be linearly increasing in time and have a datetime index.

        Parameters
        ----------
        X: pandas dataframe
            with shape [n_samples x n_features]

        Returns
        -------
        X_transformed: pandas DataFrame
            with shape [n_samples x n_features]
        """

        check_is_fitted(self, 'feature_names_')

        # we need to add time time features for the transformer to work
        X, timefeatures = self._add_time_features(X)

        # create the rolling features
        X_new = self._transformer.transform(X)

        # add original data if wanted
        if self.passthrough:
            X_new = np.hstack([X.values, X_new])

        # put in dataframe
        X_new = pd.DataFrame(X_new, columns=self.feature_names_full_, index=X.index)

        # delete time features here, as we only want to return the rollings
        X_new = X_new.drop(self.timecols, axis=1)

        return X_new

    def compute_diagnostics(self, X_train, X_test, y_train, y_test):
        """
        This function is meant to provide some insight in the performance gained by adding the
        rolling features.

        For this, it computes r-squared between y_test and predictions made for two different proxy
        models of type defined by self.estimator_type.
        The first is for the original features presented in X_train and X_test (r2_base),
        and the second is for these features including the rolling features (r2_rollings).
        It also returns the fitted predictions.

        Note!: all input must be linearly increasing in time and have a datetime index.

        Parameters
        ----------
        X_train: pandas dataframe
            with shape [n_samples x n_features].
        X_test: pandas dataframe
            with shape [n_samples x n_features].
        y_train: pandas dataframe
            with shape [n_samples x n_features].
        y_test: pandas dataframe
            with shape [n_samples]

        Returns
        -------
        r2_base: float
            r-squared for the base model (without rollings)
        r2_rollings: float
            r-squared for the model including rollings
        yhat_base: 1D array
            prediction for the base model (without rollings)
        yhat_roll: 1D array
            prediction for the model including rollings
        """

        check_is_fitted(self, 'feature_names_')

        # setup base estimation
        X_train_base, _ = self._add_time_features(X_train)
        X_test_base, _ = self._add_time_features(X_test)

        base_model = self._setup_estimator()
        base_model = base_model.fit(X_train_base, y_train)
        yhat_base = base_model.predict(X_test_base)

        # and now with rolling features
        X_train_rolling = self.transform(X_train)
        X_test_rolling = self.transform(X_test)
        X_train_rolling, _ = self._add_time_features(X_train_rolling)
        X_test_rolling, _ = self._add_time_features(X_test_rolling)

        imputer = SimpleImputer()  # required as nans are created in rolling features
        X_train_rolling = imputer.fit_transform(X_train_rolling)
        X_test_rolling = imputer.transform(X_test_rolling)

        roll_model = self._setup_estimator()
        roll_model = roll_model.fit(X_train_rolling, y_train)
        yhat_roll = roll_model.predict(X_test_rolling)

        # compute r-squareds
        r2_base = r2_score(y_test, yhat_base)
        r2_rollings = r2_score(y_test, yhat_roll)

        return r2_base, r2_rollings, yhat_base, yhat_roll
