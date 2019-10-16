import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


def scale_train_test(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler=StandardScaler()):
    '''
    Fit a transformer to the train sets to scale and later invert scaling
    of the data to a distribution on which the algorithm can learn faster.

    Parameters:
    ----------
    X_train: pd.DataFrame
        containing the training features
    X_test: pd.DataFrame
        containing the test features
    y_train: pd.Series or pd.DataFrame
        containing the train target
    y_test: pd.Series or pd.DataFrame
        containing the test target
    scaler: sklearn transformer instance
        e.g. StandardScaler(),  MinMaxScaler(), or RobustScaler()
        for available options see:
        https://scikit-learn.org/stable/auto_examples/
        preprocessing/plot_all_scaling.html

    Returns:
    -------
    X_train: pd.DataFrame
        rescaled dataframe containing the train feature samples
    X_test: pd.DataFrame
        rescaled dataframe containing the test feature samples
    y_train: pd.Series or pd.DataFrame
        rescaled series containing the train target samples
    y_train: pd.Series or pd.DataFrame
        rescaledseries containing the test target samples
    X_scaler: transformer
        fitted X transformer instance that can be used to invert scaling
    y_scaler: transformer
        fitted y transformer instance that can be used to invert scaling
    '''

    X_scaler, y_scaler = clone(scaler), clone(scaler)
    X_train = pd.DataFrame(
        X_scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index)
    X_test = pd.DataFrame(
        X_scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index)

    # we only need to test if train set is series of dataframe,
    # as this needs to be the same for the test set
    if isinstance(y_test, pd.Series):
        y_train = pd.Series(
            np.ravel(y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))),
            index=y_train.index)
        y_test = pd.Series(
            np.ravel(y_scaler.transform(np.array(y_test).reshape(-1, 1))),
            index=y_test.index)
    elif isinstance(y_train, pd.DataFrame):
        y_train = pd.DataFrame(
            y_scaler.fit_transform(y_train),
            columns=y_train.columns,
            index=y_train.index)
        y_test = pd.DataFrame(
            y_scaler.transform(y_test),
            columns=y_test.columns,
            index=y_test.index)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler
