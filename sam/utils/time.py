import pandas as pd
import numpy as np

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

    """
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
        raise ValueError("The unit is '%s', but it must start with sec, min, hour, day or week" % unit)


def label_dst(timestamps):
    """Find possible conflicts due to daylight savings time, by
    labeling timestamps. This converts a series of timestamps to
    a series of strings. The strings are either 'normal',
    'to_summertime', or 'to_wintertime'.
    to_summertime happens the last sunday morning of march,
    from 2:00 to 2:59.
    to_wintertime happens the last sunday morninig of october,
    from 2:00 to 2:59.
    These can be possible problems because they either happen 2
    or 0 times. to_summertime should therefore be impossible.

    Parameters
    ----------
    timestamps: Series, shape = (n_inputs,)
        a series of pandas timestamps

    Returns
    -------
    labels: string, array-like, shape = (n_inputs,)
        a numpy array of strings, that are all either
        'normal', 'to_summertime', or 'to_wintertime'
    """
    last_sunday_morning = (timestamps.dt.day >= 25) & \
                          (timestamps.dt.weekday == 6) & \
                          (timestamps.dt.hour == 2)
    return np.where((last_sunday_morning) & (timestamps.dt.month == 3),
                    "to_summertime",
                    np.where((last_sunday_morning) & (timestamps.dt.month == 10),
                             "to_wintertime",
                             "normal"))


def average_winter_time(data, tmpcol='tmp_UNID'):
    """Solve duplicate timestamps in wintertime, by averaging them
    Because the to_wintertime hour happens twice, there can be duplpicate timestamps
    This function removes those duplicates by averaging the VALUE column
    All other columns are used as group-by columns
    
    Parameters
    ----------
    data: pandas Dataframe
        must have columns TIME, VALUE, and optionally others like ID and TYPE.
    tmpcol: string, optional (default='tmp_UNID')
        temporary columnname that is created in dataframe. This columnname cannot
        exist in the dataframe already
    
    Returns
    -------
    data: pandas Dataframe
        The same dataframe as was given in input, but with duplicate timestamps
        removed, if they happened during the wintertime duplicate hour
    """
    assert tmpcol not in data.columns
    dst_labels = label_dst(data.TIME)
    # We make a column that is unique for all except wintertime
    # This means that in the groupby line, non-to_wintertime
    # lines will never be grouped
    data[tmpcol] = np.where(dst_labels == 'to_wintertime',
                            -1, np.arange(len(data.index)))
    groupcols = data.columns.tolist()
    groupcols.remove('VALUE')  # in place only
    data = data.groupby(groupcols).mean().reset_index().drop(tmpcol, axis=1)
    return data
