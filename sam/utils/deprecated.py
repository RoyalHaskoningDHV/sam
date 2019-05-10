import logging
import warnings
logger = logging.getLogger(__name__)


def MongoWrapper(*args, **kwargs):
    from sam.data_sources import MongoWrapper
    msg = ("sam.utils.MongoWrapper is deprecated. "
           "Please use sam.data_sources.MongoWrapper instead. "
           "sam.utils.MongoWrapper will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return MongoWrapper(*args, **kwargs)


def label_dst(*args, **kwargs):
    from sam.preprocessing import label_dst
    msg = ("sam.utils.label_dst is deprecated. Please use sam.preprocessing.label_dst instead. "
           "sam.utils.label_dst will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return label_dst(*args, **kwargs)


def average_winter_time(*args, **kwargs):
    from sam.preprocessing import average_winter_time
    msg = ("sam.utils.average_winter_time is deprecated. "
           "Please use sam.preprocessing.average_winter_time instead. "
           "sam.utils.average_winter_time will be removed in a future release.")
    warnings.warn(msg, DeprecationWarning)
    return average_winter_time(*args, **kwargs)


def unit_to_seconds(unit):
    """Performs a lookup to convert a string to a number of seconds
    the unit can be something like 'hours', or 'day'. Several options
    are supported

    Year is not supported (for now), because of leap years. They cannot
    always be uniquely converted to a number of seconds.

    Parameters
    ----------
    unit: string
        a lowercase string describing a time unit. Must start with
        sec, min, hour, day, or week

    Returns
    -------
    seconds: number
        A number describing the number of seconds that the unit takes
        For example, if unit = 'hour', then the result will be 3600.

    Examples
    --------
    >>> from sam.utils import unit_to_seconds
    >>> unit_to_seconds("week")
    604800
    """
    msg = ("unit_to_seconds is deprecated, and will be removed in a future release. "
           "Please use pandas instead, e.g. pd.Timedelta(unit).total_seconds()")
    warnings.warn(msg, DeprecationWarning)

    # fix easy issues like upper case and whitespace
    unit = unit.lower().strip()

    if unit.startswith('sec'):
        return 1
    elif unit.startswith('min'):
        return 60
    elif unit.startswith('hour'):
        return 60 * 60
    elif unit.startswith('day'):
        return 60 * 60 * 24
    elif unit.startswith('week'):
        return 60 * 60 * 24 * 7
    else:
        raise ValueError(("The unit is '%s', but it must start with "
                          "sec, min, hour, day or week") % unit)
