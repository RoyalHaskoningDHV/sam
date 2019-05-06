from sklearn.metrics import precision_recall_curve, roc_auc_score
from sam.metrics import precision_incident_recall_curve


def make_precision_recall_curve(y_true, y_score, range_pred=None, color='b', alpha=0.2):
    """
    Create and return a precision-recall curve. It does this by putting the
    precision on the y-axis, the recall on the x-axis, and for each threshold,
    plotting that precision-recall.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    y_true : array_like, shape = (n_outputs,)
        The truth values. Must be either 0 or 1. If range_pred is provided, this refers
        to the incidents.
    y_score: array_like, shape = (n_outputs,)
        The prediction. Must be between 0 and 1
    range_pred : tuple, optional (default = None)
        If this is provided, make a precision/incident recall plot, using y_true as the
        incidents, and this range_pred.
    color: string, optional (default = 'b')
        The color the the plot. Default is blue
    alpha: numeric, optional (default = 0.2)
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
    >>> from sam.visualization import make_precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3])
    >>>
    >>> make_precision_recall_curve(y_true, y_scores)

    >>> # Incident recall curve
    >>> y_incidents = np.array([0, 0, 0, 0, 1, 0])
    >>> make_precision_recall_curve(y_incidents, y_scores, (0, 2))
    """

    # shamelessly stolen from
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    import matplotlib.pyplot as plt

    if range_pred is None:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_score = roc_auc_score(y_true, y_score)
        title = '2-class Precision-Recall curve. AUC={0:0.2f}'.format(roc_score)
        recall_label = 'Recall'
    else:
        precision, recall, _ = precision_incident_recall_curve(y_true, y_score, range_pred)
        title = '2-class Precision-Incident Recall curve. Prediction range: ({}, {})'.\
            format(range_pred[0], range_pred[1])
        recall_label = 'Incident Recall'

    ax = plt.subplot()
    ax.step(recall, precision, color=color, alpha=1, where='post')
    ax.fill_between(recall, precision, alpha=alpha, color=color, step='post')

    ax.set_xlabel(recall_label)
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    return ax
