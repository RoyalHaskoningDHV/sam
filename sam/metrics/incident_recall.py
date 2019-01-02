import numpy as np


def incident_recall(y_true, y_pred, y_incidents, range_pred=(0, 0)):
    """
    Given y_true, y_pred, y_incidents and a prediction range,
    see what percentage of incidents in y_incidents was positively
    predicted in y_pred, within window range_pred.

    For use in a make_scorer, e.g.
    make_scorer(incident_recall, y_incidents=df['incidents'], range_pred=(1,5))

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
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

    """
    y_true, y_pred, y_incidents = np.array(y_true), np.array(y_pred), np.array(y_incidents)

    # Get the incides of the actual incidents
    incident_indices = np.reshape(np.nonzero(y_incidents), -1)

    # Get the ranges that a positive prediction should have been made
    # Note: we expect 3 rows to be checked when range_pred = (1,3)
    # namely 1,2,3. Top achieve this indexing, we should subtract 1 more from
    # the i-range_pred[i]
    incident_ranges = [(np.maximum(i-range_pred[1]-1, 0), np.maximum(i-range_pred[0], 0))
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
        signature is scorer(clf, X, y), where clf is a fit model,
        X is test data, and y is the ground truth for the test data
    
    """
    def incident_recall_scorer(clf, X, y):
        y_pred = clf.predict(X)
        return incident_recall(y, y_pred, X[colname], range_pred)
    return incident_recall_scorer
