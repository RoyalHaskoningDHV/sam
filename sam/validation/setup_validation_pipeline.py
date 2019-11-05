from sklearn.pipeline import Pipeline
from sam.validation import RemoveExtremeValues, RemoveFlatlines
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def create_validation_pipe(
        cols,
        rollingwindow,
        remove_extreme_values=True,
        remove_flatlines=True,
        impute_values=True,
        madthresh=15,
        flatwindow=2,
        max_iter=10,
        n_nearest_features=10,
        n_estimators=10,
        impute_method='iterative'):
    '''
    Sets up a pipeline to do data validation. Can incorporate:

    - remove extreme values
    - remove flatlines
    - impute these values

    Parameters
    ---------
    cols: list of strings
        which columns in the dataframe to apply data validation to
    rollingwindow: int or str
        parameter used in RemoveExtremeValues, see :ref:`RemoveExtremeValues`
    remove_extreme_values: bool (default=True)
        if set to True, will find extreme values and set to nan
    remove_flatlines: bool (default=True)
        if set to True, will find flatline signals
    impute_values: bool (default=True)
        if set to True, will impute found nan signals
    madthresh: int (default=15)
        parameter used in RemoveExtremeValues, see :ref:`RemoveExtremeValues`
    flatwindow: int (default=2)
        parameter used in RemoveFlatlines, see :ref:`RemoveFlatlines`
    max_iter: int (default=10)
        how many iterations to try for iterative_imputer
        see https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    n_nearest_features: int (default=10)
        how many close features to use for iterative_imputer
        `Example <https://scikit-learn.org/stable/modules/generated/
        sklearn.impute.IterativeImputer.html>`
    n_estimators: int (default=10)
        how many trees to use in RandomForestRegressor for iterative imputer
        `Example <http://www.scikit-learn.org/stable/modules/generated/
        sklearn.ensemble.RandomForestRegressor.html>`
    impute_method: string (default='iterative')
        if set to 'iterative', will impute values using IterativeImputer. This is much slower,
        but also much more accurate. Can also be set to any of the SimpleImputer strategies:
        'mean', 'median', 'most_frequent', 'constant'

    Returns
    -------
    pipe: sklearn.pipeline.Pipeline instance
        The input data should be a wide-format dataframe, where rows are time and columns are
        features.
        The rows of the data should be linearly increasing in time, and can contain gaps in time.
        However, when a string is used to specify the rollingwindow parameter, the input data
        should have a datetime index.

    Example:
    >>> from sam.validation import create_validation_pipe, validate_and_impute
    >>> from sam.visualization import diagnostic_extreme_removal, diagnostic_flatline_removal
    >>>
    >>> # create some data
    >>> np.random.seed(10)
    >>> base = np.random.randn(100)
    >>> X_train = pd.DataFrame(np.tile(base, (3, 3)).T)
    >>> X_test = pd.DataFrame(np.tile(base, (3, 1)).T)
    >>> y_test = pd.Series(base, name='target')
    >>> y_train = pd.Series(np.tile(base, 3).T, name='target')
    >>>
    >>> # add outliers to y_test:
    >>> y_test.iloc[[5, 10, 61]] *= 30
    >>> # add flatlines to y_train and y_test:
    >>> y_test.iloc[20:40] = 1
    >>> y_train.iloc[20:50] = 1
    >>>
    >>> # setup pipeline
    >>> pipe = create_validation_pipe(cols=list(X_train.columns) + ['target'], rollingwindow=5,
    >>>                              impute_method='iterative')
    >>>
    >>> # put data together
    >>> train_data = X_train.join(y_train)
    >>> test_data = X_test.join(y_test)
    >>>
    >>> # now fit the pipeline on the train data and transform both train and test
    >>> train_data = pd.DataFrame(pipe.fit_transform(train_data), columns=train_data.columns,
    >>>                           index=train_data.index)
    >>> test_data = pd.DataFrame(pipe.transform(test_data), columns=test_data.columns,
    >>>                          index=test_data.index)
    >>>
    >>> # the fitted pipeline can now be passed to diagnostic plot functions:
    >>> # create validation visualizations
    >>> f_ext = diagnostic_extreme_removal(
    >>>     pipe['extreme'], pd.DataFrame(test_data['target'], columns=['target']), 'target')
    >>> f_ext = diagnostic_flatline_removal(
    >>>     pipe['flat'], pd.DataFrame(test_data['target'], columns=['target']), 'target')
    '''

    methods = ['iterative', 'mean', 'median', 'most_frequent', 'constant']
    assert impute_method in methods, print('impute method not in %s' % methods)

    estimators = []
    if remove_extreme_values:
        REV = RemoveExtremeValues(cols=cols, rollingwindow=rollingwindow, madthresh=madthresh)
        estimators.append(['extreme', REV])

    if remove_flatlines:
        RFV = RemoveFlatlines(cols=cols, window=flatwindow)
        estimators.append(['flat', RFV])

    if impute_values:
        if impute_method == 'iterative':
            estimator = RandomForestRegressor(n_estimators=n_estimators)
            IMP = IterativeImputer(estimator=estimator, verbose=2,
                                   n_nearest_features=n_nearest_features)
        else:
            IMP = SimpleImputer(strategy=impute_method)
        estimators.append(['impute', IMP])

    pipe = Pipeline(estimators)

    return pipe
