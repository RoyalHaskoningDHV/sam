import warnings


def make_incident_heatmap(*args, **kwargs):
    from sam.visualization import plot_incident_heatmap
    msg = ("make_incident_heatmap is deprecated. Please use plot_incident_heatmap instead. "
           "make_incident_heatmap will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return plot_incident_heatmap(*args, **kwargs)


def make_precision_recall_curve(*args, **kwargs):
    from sam.visualization import plot_precision_recall_curve
    msg = ("make_precision_recall_curve is deprecated. "
           "Please use plot_precision_recall_curve instead. "
           "make_precision_recall_curve will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return plot_precision_recall_curve(*args, **kwargs)


def make_threshold_plot(*args, **kwargs):
    from sam.visualization import plot_threshold_curve
    msg = ("make_threshold_plot is deprecated. Please use plot_threshold_curve instead. "
           "make_threshold_plot will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return plot_threshold_curve(*args, **kwargs)
