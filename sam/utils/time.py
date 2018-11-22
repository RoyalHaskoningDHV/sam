
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
