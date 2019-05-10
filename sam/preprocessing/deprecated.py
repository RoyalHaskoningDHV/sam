import warnings


def complete_timestamps(*args, **kwargs):
    from sam.preprocessing import normalize_timestamps
    msg = ("complete_timestamps is deprecated. Please use normalize_timestamps instead. "
           "complete_timestamps will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return normalize_timestamps(*args, **kwargs)
