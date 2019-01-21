import numpy as np
from sam.feature_engineering import range_lag_column
from sklearn.metrics import precision_recall_curve


def incident_recall(y_pred, y_incidents, range_pred=(0, 0)):
    """
    Given y_pred, y_incidents and a prediction range,
    see what percentage of incidents in y_incidents was positively
    predicted in y_pred, within window range_pred.

    For use in a make_scorer, e.g.
    make_scorer(incident_recall, y_incidents=df['incidents'], range_pred=(1,5))

    Parameters
    ----------
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    y_incidents : 1d array-like, or label indicator array / sparse matrix
        Incidents that we want to predict with the classifier within window range_pred
    range_pred : tuple of int, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        If we want to predict an incidents 5 to 1 rows in advance, this will be (1,5)

    Returns
    -------
    result : float
        The percentage of incidents that was positively predicted

    Examples
    --------
    >>> from sam.metrics import incident_recall
    >>> y_pred = [1,0,0,1,0,0,0]
    >>> y_incidents = [0,1,0,0,0,0,1]
    >>> range_pred = (0,2)
    >>> incident_recall(y_pred, y_incidents, range_pred)
    0.5
    """
    assert range_pred[0] >= 0 and range_pred[1] >= 0, "prediction window must be positive"
    y_pred, y_incidents = np.array(y_pred), np.array(y_incidents)

    # Get the incides of the actual incidents
    incident_indices = np.reshape(np.nonzero(y_incidents), -1)

    # Get the ranges that a positive prediction should have been made
    # Note: we expect 3 rows to be checked when range_pred = (1,3)
    # namely 1,2,3. Top achieve this indexing, we should add 1 to
    # the i-range_pred[0] to make the range inclusive of the last record
    incident_ranges = [(np.maximum(i-range_pred[1], 0), np.maximum(i-range_pred[0]+1, 0))
                       for i in incident_indices]

    # Find out if there's any positive prediction in this range
    incidents_found = [np.any(y_pred[start:end]) for start, end in incident_ranges]

    # Calculate the score
    score = np.sum(incidents_found) / len(incident_indices)

    return score


def make_incident_recall_scorer(range_pred=(0, 0), colname='incident'):
    """
    Wrapper around incident_recall_score, to make it an actual sklearn scorer.
    This works by obtaining the 'incident' column from the data itself. This
    column name is configurable. This scorer does need the incident column
    to be present, which means by default it will be used in training the model.
    If this is not desired, a custom model will have to be made that deletes
    this variable before fitting/predicting.

    Parameters
    ----------
    range_pred : tuple of int, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        Passed through to incident_recall
    colname : string, optional (default = "indicent")
        The column name that is given to incident_recall as incidents
        we want to predict

    Returns
    -------
    scorer : a function that acts as a sklearn scorer object.
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
    >>>          {"predict": lambda self, X: np.array([0, 1, 0, 0, 0, 0, 0, 0])})
    >>> data = pd.DataFrame({"incident": [0, 0, 0, 1, 1, 0, 1, 0], "other": 1})
    >>>
    >>> scorer = make_incident_recall_scorer((1, 3), "incident")
    >>> scorer(op(), data)
    0.66666
    """
    def incident_recall_scorer(clf, X):
        y_pred = clf.predict(X)
        return incident_recall(y_pred, X[colname], range_pred)
    return incident_recall_scorer


def precision_incident_recall_curve(y_incidents, y_pred, range_pred=(0, 0)):
    """
    Analogous to sklearn.metrics.precision_recall_curve, but for incident recall and
    precision.
    Precision is what it seems: for every prediction you make, check if there
    really is an incident coming up or not. Precision is the percentage you get correct.
    Incident recall is as in incident_recall function: for every incident, check if you
    made at least a single positive prediction.

    The calculation of the thresholds is done by calling sklearn.precision_recall_curve.

    Given incidents and a prediction, as well as range,
    returns precision, recall, thresholds.

    Parameters
    ----------
    y_incidents : 1d array-like, or label indicator array / sparse matrix
        Incidents that we want to predict with the classifier within window range_pred
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    range_pred : tuple of int, optional (default = (0,0))
        length of two: (start prediction window, end prediction window)
        If we want to predict an incidents 5 to 1 rows in advance, this will be (1,5)

    Returns
    -------
    precision: 1d array-like: an array of precision scores
    recall: 1d array-like: an array of incident recall scores
    thresholds: 1d array-like: an array of thresholds to interpret the above 2

    Examples
    --------
    >>> y_pred = [0.4, 0.4,  0.1, 0.2, 0.6,  0.5, 0.1]
    >>> y_incidents = [  0, 1, 0, 0, 0, 0, 1]
    >>> range_pred = (0, 2)
    >>> p, r, t = incident_precision_recall_curve(y_incidents, y_pred, range_pred)
    >>> p
    array([0.71428571, 0.8, 1., 1., 1., 1.])
    >>> r
    array([1., 1., 1., 0.6, 0.6, 0.])
    >>> t
    array([0.1, 0.2, 0.4, 0.5, 0.6])
    """
    y_lagged = range_lag_column(y_incidents, range_pred)
    precision, _, thresholds = precision_recall_curve(y_lagged, y_pred)
    recall = [incident_recall(y_pred > t, y_incidents, range_pred) for t in thresholds]
    # The first element is missing due to the way thresholds is always 1 shorter than recall.
    # However, we are missing the first value, which is by definition always 1, so it's easy to add
    recall = np.array([1] + recall)
    return precision, recall, thresholds
