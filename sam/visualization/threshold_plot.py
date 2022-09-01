import numpy as np
from sam.metrics import precision_incident_recall_curve
from sklearn.metrics import precision_recall_curve


def plot_threshold_curve(y_true: np.array, y_score: np.array, range_pred: tuple = None):
    """
    Create and return a threshold curve, also known as Ynformed plot. It does this by putting
    the threshold on the x-axis, and for each threshold, plotting the precision and recall.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    y_true: array_like, shape = (n_outputs,)
        The actual values. Values must be either 0 or 1. If range_pred is provided, this refers
        to the incidents.
    y_score: array_like, shape = (n_outputs,)
        The prediction. Values must be between 0 and 1
    range_pred: tuple, optional (default = None)
        If provided, make a precision/incident recall plot, using y_true as the
        incidents, and this range_pred.

    Returns
    -------
    plot:  matplotlib.axes._subplots.AxesSubplot object
        a plot containing the resulting precision threshold plot.
        can be edited further, or printed to the output.

    Examples
    --------
    >>> from sam.visualization import plot_threshold_curve
    >>> plot_threshold_curve([0, 1, 0, 1, 1, 0], [.2, .3, .4, .5, .9, .1])  # doctest: +SKIP

    >>> # Incident recall threshold plot
    >>> plot_threshold_curve(
    ... [0, 0, 0, 0, 1, 0], [.2, .3, .4, .5, .9, .1], (0, 1)
    ... )  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if range_pred is None:
        # Retrieve the curve data
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        recall_label = "Recall"
    else:
        precision, recall, thresholds = precision_incident_recall_curve(
            y_true, y_score, range_pred
        )
        recall_label = "Incident Recall"

    # Initialize the figure
    _, ax = plt.subplots()
    ax.set_title("Precision and Recall Scores per threshold")

    # Plot the curves in the figure
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label=recall_label)
    ax.set_ylabel("Score")
    ax.set_xlabel("Decision Threshold")
    ax.legend(loc="best")

    return ax
