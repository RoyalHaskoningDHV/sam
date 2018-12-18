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
    range_pred : tuple (start prediction window, end prediction window), default (0,0)
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
    incident_ranges = [(np.maximum(i-range_pred[1], 0), np.maximum(i-range_pred[0], 0))
                       for i in incident_indices]

    # Find out if there's any positive prediction in this range
    incidents_found = [np.any(y_pred[start:end]) for start, end in incident_ranges]

    # Calculate the score
    score = np.sum(incidents_found) / len(incident_indices)

    return score
