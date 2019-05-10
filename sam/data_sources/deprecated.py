import warnings


def create_synthetic_timeseries(*args, **kwargs):
    from sam.data_sources import synthetic_timeseries
    msg = ("create_synthetic_timeseries is deprecated. Please use synthetic_timeseries instead. "
           "create_synthetic_timeseries will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return synthetic_timeseries(*args, **kwargs)
