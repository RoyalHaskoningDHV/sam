from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt


def make_precision_recall_curve(y_true, y_score, color='b', alpha=0.2):
    """
    Create and return a precision-recall curve. It does this by putting the
    precision on the y-axis, the recall on the x-axis, and for each threshold,
    plotting that precision-recall.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    y_true : array_like, shape = (n_outputs,)
        The truth values. Must be either 0 or 1
    y_score: array_like, shape = (n_outputs,)
        The prediction. Must be between 0 and 1
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
    """

    # shamelessly stolen from
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_score = roc_auc_score(y_true, y_score)
    ax = plt.subplot()
    ax.step(recall, precision, color=color, alpha=1, where='post')
    ax.fill_between(recall, precision, alpha=alpha, color=color, step='post')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('2-class Precision-Recall curve. ROC={0:0.2f}'.format(roc_score))
    return ax