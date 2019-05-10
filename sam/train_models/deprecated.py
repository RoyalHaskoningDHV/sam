import warnings


def find_outlier_curves(*args, **kwargs):
    from sam.exploration import incident_curves
    msg = ("sam.train_models.find_outlier_curves is deprecated. "
           "Please use sam.exploration.incident_curves instead. "
           "sam.train_models.find_outlier_curves will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return incident_curves(*args, **kwargs)


def create_outlier_information(*args, **kwargs):
    from sam.exploration import incident_curves_information
    msg = ("sam.train_models.create_outlier_information is deprecated. "
           "Please use sam.exploration.incident_curves_information instead. "
           "sam.train_models.create_outlier_information will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return incident_curves_information(*args, **kwargs)
