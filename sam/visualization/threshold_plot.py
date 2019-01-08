from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def make_threshold_plot(y_true, y_score):
    """
    Create and return a threshold, also Ynformed, plot. It does this by putting
    the threshold on the x-axis, and for each threshold, plotting
    the precision and recall.
    This returns a subplot object that can be shown or edited further.

    Parameters
    ----------
    y_true : array_like, shape = (n_outputs,)
        The truth values. Must be either 0 or 1
    y_score: array_like, shape = (n_outputs,)
        The prediction. Must be between 0 and 1

    Returns
    -------
    plot:  matplotlib.axes._subplots.AxesSubplot object
        a plot containing the resulting precision threshold plot.
        can be edited further, or printed to the output.

    Examples
    --------
    >>> from sam.visualization import make_threshold_plot
    >>> make_threshold_plot([0, 1, 0, 1, 1, 0],[.2, .3, .4, .5, .9, .1])
    <matplotlib.axes._subplots.AxesSubplot at 0x7f67d1ea4630>
    """

    # Retrieve the curve data
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Initialize the figure
    fig, ax = plt.subplots()
    ax.set_title("Precision and Recall Scores per threshold")

    # Plot the curves in the figure
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.set_ylabel("Score")
    ax.set_xlabel("Decision Threshold")
    ax.legend(loc='best')

    return ax
