import numpy as np
from sam.metrics import precision_incident_recall_curve
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_precision_recall_curve(
    y_true: np.array,
    y_score: np.array,
    range_pred: tuple = None,
    color: str = "b",
    alpha: float = 0.2,
):
    """
    Create and return a precision-recall curve. It does this by putting the
    precision on the y-axis, the recall on the x-axis, and for each threshold,
    plotting that precision-recall.
    The function returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    y_true : np.array, shape = (n_outputs,)
        The truth values. Must be either 0 or 1. If range_pred is provided, this refers
        to the incidents.
    y_score: np.array, shape = (n_outputs,)
        The prediction. Must be between 0 and 1
    range_pred : tuple, optional (default = None)
        If this is provided, make a precision/incident recall plot, using y_true as the
        incidents, and this range_pred.
    color: string, optional (default='b')
        The color the the plot. Default is blue
    alpha: float, optional (default=0.2)
        The transparency of the solid part of the plot.
        The line will always be alpha 0.1, so if you
        just want the line, set this to 0.

    Returns
    -------
    plot:  matplotlib.axes._subplots.AxesSubplot object
        a plot containing the resulting precision recall_curve.
        can be edited further, or printed to the output.

    Examples
    --------
    >>> from sam.visualization import plot_precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3])
    >>>
    >>> fig = plot_precision_recall_curve(y_true, y_scores)

    >>> # Incident recall curve
    >>> y_incidents = np.array([0, 0, 0, 0, 1, 0])
    >>> fig2 = plot_precision_recall_curve(y_incidents, y_scores, (0, 2))
    """
    import matplotlib.pyplot as plt

    if range_pred is None:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        title = "2-class Precision-Recall curve. AP={0:0.2f}".format(ap_score)
        recall_label = "Recall"
    else:
        precision, recall, _ = precision_incident_recall_curve(y_true, y_score, range_pred)
        title = "2-class Precision-Incident Recall curve. Prediction range: ({}, {})".format(
            range_pred[0], range_pred[1]
        )
        recall_label = "Incident Recall"

    plt.figure()
    ax = plt.subplot()
    ax.step(recall, precision, color=color, alpha=1, where="post")
    ax.fill_between(recall, precision, alpha=alpha, color=color, step="post")

    ax.set_xlabel(recall_label)
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    return ax
