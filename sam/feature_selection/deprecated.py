import warnings


def retrieve_top_n_correlations(*args, **kwargs):
    from sam.exploration import top_n_correlations
    msg = ("sam.feature_selection.retrieve_top_n_correlations is deprecated. "
           "Please use sam.exploration.top_n_correlations instead. "
           "sam.feature_selection.retrieve_top_n_correlations "
           "will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return top_n_correlations(*args, **kwargs)


def retrieve_top_score_correlations(*args, **kwargs):
    from sam.exploration import top_score_correlations
    msg = ("sam.feature_selection.retrieve_top_score_correlations is deprecated. "
           "Please use sam.exploration.top_score_correlations instead. "
           "sam.feature_selection.retrieve_top_score_correlations "
           "will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return top_score_correlations(*args, **kwargs)


def create_lag_correlation(*args, **kwargs):
    from sam.exploration import lag_correlation
    msg = ("sam.feature_selection.create_lag_correlation is deprecated. "
           "Please use sam.exploration.lag_correlation instead. "
           "sam.feature_selection.create_lag_correlation will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return lag_correlation(*args, **kwargs)
