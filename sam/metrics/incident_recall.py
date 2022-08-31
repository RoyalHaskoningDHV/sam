from typing import Tuple

import numpy as np
import pandas as pd
from sam.feature_engineering import range_lag_column
from sklearn.metrics import precision_recall_curve


def incident_recall(
    y_incidents: np.ndarray,
    y_pred: np.ndarray,
    range_pred: Tuple[int, int] = (0, 0),
):
    """
    Given `y_pred`, `y_incidents` and a prediction range, see what percentage of incidents in
    `y_incidents` was positively predicted in `y_pred`, within window `range_pred`. Works for
    binary classification only (predicting 1 means incident, predicting 0 means no incident)

    For use in a make_scorer, e.g.
    `make_scorer(incident_recall, y_incidents=df['incidents'], range_pred=(1,5))`

    Parameters
    ----------
    y_incidents: 1d array-like, or label indicator array / sparse matrix
        Incidents that we want to predict with the classifier within window range_pred
    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    range_pred: tuple of ints, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        If we want to predict an incidents 5 to 1 rows in advance, this will be (1,5)
        range_pred is inclusive, so (0, 1) means you can predict either 0 or 1
        timesteps in advance

    Returns
    -------
    result: float
        The percentage of incidents that was positively predicted

    Examples
    --------
    >>> from sam.metrics import incident_recall
    >>> y_pred = [1,0,0,1,0,0,0]
    >>> y_incidents = [0,1,0,0,0,0,1]
    >>> range_pred = (0,2)
    >>> incident_recall(y_incidents, y_pred, range_pred)
    0.5
    """
    if range_pred[0] < 0 or range_pred[1] < 0:
        raise ValueError("Prediction window range_pred must be positive")
    y_pred, y_incidents = pd.Series(y_pred), pd.Series(y_incidents)
    # A prediction has effect on the future, so lag to the future
    y_pred = range_lag_column(y_pred, (-1 * range_pred[0], -1 * range_pred[1]))
    predicted_incidents = pd.concat([y_pred, y_incidents], axis=1).min(axis=1).sum()
    return predicted_incidents / y_incidents.sum()


def make_incident_recall_scorer(
    range_pred: Tuple[int, int] = (0, 0),
    colname: str = "incident",
):
    """
    Wrapper around `incident_recall_score`, to make it an actual sklearn scorer.
    This works by obtaining the 'incident' column from the data itself. This
    column name is configurable. This scorer does need the incident column
    to be present, which means by default it will be used in training the model.
    If this is not desired, a custom model will have to be made that deletes
    this variable before fitting/predicting.

    Parameters
    ----------
    range_pred: tuple of int, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        Passed through to incident_recall
    colname: string, optional (default = "indicent")
        The column name that is given to incident_recall as incidents
        we want to predict

    Returns
    -------
    scorer: a function that acts as a sklearn scorer object.
        signature is scorer(clf, X), where clf is a fit model,
        X is test data

    Examples
    --------
    >>> from sam.metrics import make_incident_recall_scorer
    >>> from sklearn.base import BaseEstimator
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> op = type("MyClassifier", (BaseEstimator, object),
    ...          {"predict": lambda self, X: np.array([0, 1, 0, 0, 0, 0, 0, 0])})
    >>> data = pd.DataFrame({"incident": [0, 0, 0, 1, 1, 0, 1, 0], "other": 1})
    >>>
    >>> scorer = make_incident_recall_scorer((1, 3), "incident")
    >>> scorer(op(), data)
    0.6666666666666666
    """

    def incident_recall_scorer(clf, X):
        y_pred = clf.predict(X)
        return incident_recall(X[colname], y_pred, range_pred)

    return incident_recall_scorer


def _merge_thresholds(
    left_t: np.ndarray,
    right_t: np.ndarray,
    left_val: np.ndarray,
    right_val: np.ndarray,
):
    """
    Helper function that merges two different thresholds. Does this by iterating over the
    thresholds, and selecting the lowest threshold as the next.
    """

    def step_ahead(new_t, new_val, saved_val, ix, old_t, old_val):
        new_t.append(old_t[ix])
        new_val.append(old_val[ix])
        ix += 1
        saved_val = old_val[ix]
        return new_t, new_val, saved_val, ix

    left_ix, right_ix = 0, 0
    new_t = []
    new_leftval, new_rightval = [], []
    saved_leftval, saved_rightval = left_val[0], right_val[0]

    while left_ix < len(left_t) or right_ix < len(right_t):
        if len(left_t) > 0 and (right_ix == len(right_t) or left_t[left_ix] < right_t[right_ix]):
            new_t, new_leftval, saved_leftval, left_ix = step_ahead(
                new_t, new_leftval, saved_leftval, left_ix, left_t, left_val
            )
            new_rightval.append(saved_rightval)

        elif len(right_t) > 0 and (left_ix == len(left_t) or left_t[left_ix] > right_t[right_ix]):
            new_t, new_rightval, saved_rightval, right_ix = step_ahead(
                new_t, new_rightval, saved_rightval, right_ix, right_t, right_val
            )
            new_leftval.append(saved_leftval)

        elif left_t[left_ix] == right_t[right_ix]:
            new_t, new_leftval, saved_leftval, left_ix = step_ahead(
                new_t, new_leftval, saved_leftval, left_ix, left_t, left_val
            )
            new_t, new_rightval, saved_rightval, right_ix = step_ahead(
                new_t, new_rightval, saved_rightval, right_ix, right_t, right_val
            )
            new_t = new_t[:-1]

    new_leftval.append(left_val[-1])
    new_rightval.append(right_val[-1])
    return np.array(new_leftval), np.array(new_rightval), np.array(new_t)


def precision_incident_recall_curve(
    y_incidents: np.ndarray, y_pred: np.ndarray, range_pred: Tuple[int, int] = (0, 0)
):
    """
    Analogous to `sklearn.metrics.precision_recall_curve
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html>`,
    but for incident recall and precision.
    Precision the percentage of correct incidents in the prediction.
    Incident recall can be found in the incident_recall function: for every incident, check if at
    least there is a single positive prediction.

    The calculation of the thresholds is done by calling `sklearn.precision_recall_curve`.

    Given incidents and a prediction, as well as range, returns precision, recall, thresholds.

    Parameters
    ----------
    y_incidents: 1d array-like, or label indicator array / sparse matrix
        Incidents that we want to predict with the classifier within window range_pred
    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    range_pred: tuple of int, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        If we want to predict an incidents 5 to 1 rows in advance, this will be (1,5)

    Returns
    -------
    precision: 1d array-like
        an array of precision scores
    recall: 1d array-like
        an array of incident recall scores
    thresholds: 1d array-like
        an array of thresholds to interpret the above 2

    Examples
    --------
    >>> y_pred = [0.4, 0.4,  0.1, 0.2, 0.6,  0.5, 0.1]
    >>> y_incidents = [  0, 1, 0, 0, 0, 0, 1]
    >>> range_pred = (0, 2)
    >>> p, r, t = precision_incident_recall_curve(y_incidents, y_pred, range_pred)
    >>> p
    array([0.71428571, 0.8       , 1.        , 1.        , 1.        ,
           1.        ])
    >>> r
    array([1. , 1. , 1. , 0.5, 0.5, 0. ])
    >>> t
    array([0.1, 0.2, 0.4, 0.5, 0.6])
    """
    if range_pred[0] < 0 or range_pred[1] < 0:
        raise ValueError("prediction window range_pred must be positive")

    y_lagged = range_lag_column(y_incidents, range_pred)
    precision, _, thresholds_p = precision_recall_curve(y_lagged, y_pred)

    y_pred_incidents = range_lag_column(y_pred, (-1 * range_pred[0], -1 * range_pred[1]))
    _, recall, thresholds_r = precision_recall_curve(y_incidents, y_pred_incidents)

    return _merge_thresholds(thresholds_p, thresholds_r, precision, recall)
