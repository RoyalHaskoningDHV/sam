from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sam.feature_engineering import BuildRollingFeatures, decompose_datetime
from sam.feature_engineering.base_feature_engineering import IdentityFeatureEngineer
from sam.utils import has_strictly_increasing_index
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

INPUT_VALIDATION_ERR_MESSAGE = (
    "all input must be linearly increasing in time and have a datetime index"
)


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
    n_iter_per_param: int (default=25)
        number of random values to try for each parameter. The total number of iterations is
        given by n_iter_per_param * len(window_sizes) * len(rolling_types)
    cv: int (default=3)
        number of cross-validated tries to attempt for each parameter combination
    estimator_type: str (default='lin')
        type of estimator to determine rolling importance. Can be one of: ['rf', 'lin', 'bayeslin']
    passthrough: bool (default=True)
        whether to pass original features in the transform method or not
    cyclicals: list (default=None)
        A list of pandas datetime properties, such as ['minute', 'hour', 'dayofweek', 'week'],
        that will be converted to cyclicals.
        The rationale here is that if time features are not added, the rolling engineering
        will find values for instance of 1 day ago to predict well, while actually this is simply
        a recurring daily pattern that can be captured by time features.
        Note that if timefeatures are added, they are not added in the transform method. Therefore,
        you will have to add them yourself during subsequent model building stages.
    onehots: list (default=None)
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
    ...     '2018-01-01',
    ...     '2019-01-01',
    ...     variables = ['T', 'FH', 'FF', 'FX', 'SQ', 'Q', 'DR', 'RH']).set_index(['TIME'])

    >>> # let's predict temperature 12 hours into the future
    >>> target = 'T'
    >>> fut = 12
    >>> y = data[target].shift(-fut).iloc[:-fut]
    >>> X = data.iloc[:-fut]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    >>> # do the feature selection, first try without adding timefeatures
    >>> ARE = AutomaticRollingEngineering(
    ...     window_sizes=[randint(1,24)], rolling_types=['mean', 'lag'])
    >>> ARE = ARE.fit(X_train, y_train)  # doctest: +ELLIPSIS
    Fitting ...

    >>> # check some diagnostics, note: results may vary as it is a random search
    >>> r2_base, r2_rollings, yhat_base, yhat_roll = ARE.compute_diagnostics(
    ...     X_train, X_test, y_train, y_test)
    >>> print(r2_base, r2_rollings)
    0.34353081601421964 0.7027068150592539

    >>> # you can also inspect feature importances:
    >>> barplot = sns.barplot(data=ARE.feature_importances_, y='feature_name', x='coefficients')

    >>> # and make plot of the timeseries:
    >>> timeseries_fig = plt.figure(figsize=(12, 6))
    >>> timeseries_fig = plt.plot(X_test.index, y_test.ravel(), 'ok', label='data')
    >>> timeseries_fig = plt.plot(
    ...     X_test.index, yhat_base, lw=3, alpha=0.75, label='yhat_base (r2: %.2f)'%r2_base
    ... )
    >>> timeseries_fig = plt.plot(
    ...     X_test.index, yhat_roll, lw=3, alpha=0.75, label='yhat_rolling (r2: %.2f)'%r2_rollings)
    >>> timeseries_fig = plt.legend(loc='best')
    """

    def __init__(
        self,
        window_sizes: List[List],
        rolling_types: List[str] = ["mean", "lag"],
        n_iter_per_param: int = 25,
        cv: int = 3,
        estimator_type: str = "lin",
        passthrough: bool = True,
        cyclicals: Optional[List[str]] = None,
        onehots: Optional[List[str]] = None,
    ):
        self.window_sizes = window_sizes
        self.rolling_types = rolling_types
        self.n_iter_per_param = n_iter_per_param
        self.cv = cv
        self.estimator_type = estimator_type
        self.passthrough = passthrough

        if cyclicals is None:
            self.cyclicals = []
        else:
            self.cyclicals = cyclicals

        if onehots is None:
            self.onehots = []
        else:
            self.onehots = onehots

    def _setup_estimator(self) -> RegressorMixin:
        if self.estimator_type == "rf":
            estimator = RandomForestRegressor(n_estimators=100, min_samples_split=5)
        elif self.estimator_type == "lin":
            estimator = LinearRegression()
        elif self.estimator_type == "bayeslin":
            estimator = BayesianRidge()

        return estimator

    def _setup_pipeline(
        self,
        original_features: List[str],
        time_features: List[str],
    ) -> Pipeline:
        """
        Create the pipeline that is later fitted. Includes:
        - BuildRollingFeatures
        - SimpleImputer
        - Model (depending on self.estimator_type)

        Parameters
        ----------
        original_features : list of strings
            list of original feature names
        time_features : list of strings
            list of time feature names

        Returns
        -------
        Pipeline
            resulting pipeline that can be fitted
        """

        rolls = []
        for feature in original_features:
            for i in range(len(self.window_sizes)):
                # only keep originals once to prevent doubling
                rolls.append(
                    (
                        "%s_%d" % (feature, i),
                        BuildRollingFeatures(keep_original=(i == 0)),
                        [feature],
                    )
                )

        for time_feature in time_features:
            rolls.append((time_feature, IdentityFeatureEngineer(), [time_feature]))

        pipeline = Pipeline(
            steps=[
                ("rollpipe", ColumnTransformer(rolls, n_jobs=-1)),
                ("imputer", SimpleImputer()),
                ("model", self._setup_estimator()),
            ]
        )

        return pipeline

    def _setup_rolling_gridsearch_params(
        self, original_features: List[str]
    ) -> Dict[str, List[Union[str, List]]]:
        """
        Create grid of rolling feature parameters

        Parameters
        ----------
        original_features : list of strings
            list of feature names

        Returns
        -------
        dict
            dictionary with entry for each window_size and rolling_type combination for each
            original feature
        """

        param_grid = {}
        for i in range(len(self.window_sizes)):
            for feature in original_features:
                param_grid["rollpipe__%s_%d__window_size" % (feature, i)] = self.window_sizes[i]
                param_grid["rollpipe__%s_%d__rolling_type" % (feature, i)] = self.rolling_types

        return param_grid

    def _validate_params(self, X_shape: tuple) -> None:
        """
        Validate the self parameters

        Parameters
        ----------
        X_shape : tuple
            tuple of array dimensions
        """

        for window_size in self.window_sizes:
            if type(window_size) is list:
                interval: list = [np.min(window_size), np.max(window_size)]
            else:
                # otherwise we assume it is a scipy.stats distribution
                interval: tuple = window_size.interval(1)

            if interval[0] < 0:
                raise ValueError("window_size must be greater than 0, but is %d" % interval[0])

            if X_shape[0] <= interval[1]:
                raise ValueError(
                    "number of samples too small for maximum window_size "
                    + "(should be smaller than X.shape[0])"
                )

        if "ewm" in self.rolling_types:
            raise ValueError("rolling_type cannot be ewm")
        if self.n_iter_per_param <= 0:
            raise ValueError("n_iter_per_param must be greater than 0")
        if self.cv <= 1:
            raise ValueError("cv must be greater than 1")

    def _add_time_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Add the timefeatures given in 'components' to X

        Parameters
        ----------
        X: pandas dataframe
            with time index and other features as columns

        Returns
        -------
        X: pandas dataframe
            original X dataframe with the time features added
        timecols: list of strings
            list of time feature names
        """

        components: list = self.cyclicals + self.onehots
        if len(components) > 0:
            times_df = pd.DataFrame({"TIME": X.index})
            time_features: pd.DataFrame = decompose_datetime(
                times_df,
                components=components,
                onehots=self.onehots,
                cyclicals=self.cyclicals,
                keep_original=True,
            )
            time_features = time_features.set_index("TIME", drop=True)

            # some columns can be constant (i.e. higher res than data is in). However, removing
            # all constant columns will also remove one-hot-encoded features that are not
            # currently in the dataset (like week_52 if you only have partial year data).
            # So, we simply leave all constant features in there.

            X = X.join(time_features)
            timecols: list = time_features.columns
        else:
            timecols = []

        return X, timecols

    def get_feature_names_out(self, input_features=None) -> List[str]:
        check_is_fitted(self, "feature_names_")
        return self.feature_names_

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
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

        if not has_strictly_increasing_index(X):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)
        if not has_strictly_increasing_index(y):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)

        # save feature names here before adding timefeatures
        original_features: list = X.columns

        # add time time features
        X, timefeatures = self._add_time_features(X)

        self._validate_params(np.shape(X))

        # setup pipeline
        pipeline: Pipeline = self._setup_pipeline(original_features, timefeatures)
        param_grid: dict = self._setup_rolling_gridsearch_params(original_features)

        n_iter: int = self.n_iter_per_param * len(self.window_sizes) * len(self.rolling_types)

        # find best rollings using randomsearch (future improvement: use BayesSearchCV from skopt)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_jobs=-1,
            cv=TimeSeriesSplit(self.cv),
            verbose=2,
            n_iter=n_iter,
        )
        search.fit(X, y)

        # recover feature names
        feature_names: list = search.best_estimator_["rollpipe"].get_feature_names_out()

        # fix names from e.g. 'RH_1__RH#mean_96' to 'RH#mean_96'
        feature_names: list = [f.split("__")[1] for f in feature_names]
        self.rolling_feature_names_: list = [f for f in feature_names if "#" in f]

        # convert these names to a set of BuildRollingFeatures that creates these features
        transformers = []
        for name in self.rolling_feature_names_:
            feature_name: str = name.split("#")[0]
            rolling_type: str = name.split("#")[1].split("_")[0]
            window_size = int(name.split("_")[-1])

            transformers.append(
                [
                    "%s_%s_%d" % (feature_name, rolling_type, window_size),
                    BuildRollingFeatures(
                        rolling_type=rolling_type,
                        window_size=window_size,
                        keep_original=False,
                    ),
                    [feature_name],
                ]
            )

        # now save the columntransformer to the self object to be used in the transform method
        self._transformer = ColumnTransformer(transformers, n_jobs=-1)
        self._transformer = self._transformer.fit(X)

        _ = search.best_estimator_[:-1].transform(X)
        if self.estimator_type == "rf":
            imp_name = "importances"
            importances: np.ndarray = search.best_estimator_["model"].feature_importances_
        elif self.estimator_type == "lin":
            imp_name = "coefficients"
            importances: np.ndarray = search.best_estimator_["model"].coef_

        # also save the feature importances under convenient name:
        self.feature_importances_ = pd.DataFrame(
            {imp_name: importances, "feature_name": feature_names}
        ).sort_values(by=imp_name, ascending=False)

        # this is required for setup of df including timefeatures in transform
        if self.passthrough:
            self.feature_names_full_: list = feature_names
        else:
            self.feature_names_full_: list = self.rolling_feature_names_

        # these are the final feature names
        self.timecols: list = [f for f in self.feature_names_full_ if "TIME" in f]
        self.feature_names_: list = [f for f in self.feature_names_full_ if "TIME" not in f]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        if not has_strictly_increasing_index(X):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)

        check_is_fitted(self, "feature_names_")

        # we need to add time time features for the transformer to work
        X, _ = self._add_time_features(X)

        # create the rolling features
        X_new: np.ndarray = self._transformer.transform(X)

        # add original data if wanted
        if self.passthrough:
            X_new = np.hstack([X.values, X_new])

        # put in dataframe
        X_new = pd.DataFrame(X_new, columns=self.feature_names_full_, index=X.index)

        # delete time features here, as we only want to return the rollings
        X_new = X_new.drop(self.timecols, axis=1)

        return X_new

    def compute_diagnostics(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
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

        if not has_strictly_increasing_index(X_train):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)
        if not has_strictly_increasing_index(X_test):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)
        if not has_strictly_increasing_index(y_train):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)
        if not has_strictly_increasing_index(y_test):
            raise ValueError(INPUT_VALIDATION_ERR_MESSAGE)

        check_is_fitted(self, "feature_names_")

        # setup base estimation
        X_train_base, _ = self._add_time_features(X_train)
        X_test_base, _ = self._add_time_features(X_test)

        base_model: RegressorMixin = self._setup_estimator()
        base_model = base_model.fit(X_train_base, y_train)
        yhat_base: np.ndarray = base_model.predict(X_test_base)

        # and now with rolling features
        X_train_rolling: pd.DataFrame = self.transform(X_train)
        X_test_rolling: pd.DataFrame = self.transform(X_test)
        X_train_rolling, _ = self._add_time_features(X_train_rolling)
        X_test_rolling, _ = self._add_time_features(X_test_rolling)

        imputer = SimpleImputer()  # required as nans are created in rolling features
        X_train_rolling = imputer.fit_transform(X_train_rolling)
        X_test_rolling: pd.DataFrame = imputer.transform(X_test_rolling)

        roll_model: RegressorMixin = self._setup_estimator()
        roll_model = roll_model.fit(X_train_rolling, y_train)
        yhat_roll: np.ndarray = roll_model.predict(X_test_rolling)

        # compute r-squareds
        r2_base: float = r2_score(y_test, yhat_base)
        r2_rollings: float = r2_score(y_test, yhat_roll)

        return r2_base, r2_rollings, yhat_base, yhat_roll
