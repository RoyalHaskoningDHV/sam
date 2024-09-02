import numpy as np
import pandas as pd


def _interpolate_pattern(bigtime, smalltime=None, pattern=0, length=1):
    """
    Helper function to create a cubic spline
    The magnitude of the cubic spline is given by pattern
    The x-values of the cubic spline are given by bigtime
    Length is the number of different values of bigtime
    For example:
    _interpolate_pattern(date.dt.month, date.dt.day, 5, 12)
    Will create a cubic spline between points on the first day of the month
    """
    if isinstance(pattern, (int, float)):
        pattern = np.random.uniform(0, pattern, length)
    else:
        for value in pattern:
            if not isinstance(value, (int, float)):
                raise TypeError("pattern must be a list of int or float")

    spline = np.full(bigtime.size, np.nan)
    for ix, value in enumerate(pattern):
        if smalltime is None:
            spline[(bigtime == ix)] = value
        else:
            spline[(bigtime == ix) & (smalltime == 0)] = value
    try:
        return pd.Series(spline).interpolate("cubic").values
    except ValueError:
        # numeric nonsense, probably happens only if the length of bigtime is too small
        return np.zeros(bigtime.size)


def _add_temporal_noise(time, noisetype="poisson", noisesize=0, length=1):
    """
    Helper function to create noise that is different but predictable
    The noise must be poisson or normal
    For example, to make some days of the week noisier than other days
    _add_temporal_noise(date.dt.month, 'normal', 2, 12)
    will generate 12 numbers between 0 and 2. Then, it will add gaussian
    noise to each month, with magnitude of the generated number.
    """
    if isinstance(noisesize, (int, float)):
        noisesize = np.random.poisson(noisesize, length)
    temp = np.zeros(time.size)
    for ix, value in enumerate(noisesize):
        if noisetype == "poisson":
            temp[time == ix] = np.random.poisson(lam=value, size=temp[time == ix].size)
        if noisetype == "normal":
            temp[time == ix] = np.random.normal(loc=0, scale=value, size=temp[time == ix].size)
    return temp


def synthetic_timeseries(
    dates,
    monthly=0,
    daily=0,
    hourly=0,
    monthnoise=(None, 0),
    daynoise=(None, 0),
    noise={},
    minmax_values=None,
    cutoff_values=None,
    negabs=None,
    random_missing=None,
    seed=None,
):
    """
    Create a synthetic time series, with some temporal patterns, and some noise. There are various
    parameters to control the distribution of the variables. The output will never be completely
    realistic, it will at least resemble what real life data could look like.

    The algorithm works like this:

    - 3 cubic splines are created: one with a monthly pattern, one with a day-of-week pattern, and
      one with an hourly pattern. These splines are added together.
    - For each month and day of the week, noise is generated according to monthnoise and daynoise
      These two sources of noise are added together
    - Noise as specified by the noise parameter is generated for each point
    - The above three series are added together. Rescale the result according to `minmax_values`
    - Missing values are added according to `cutoff_values` and `random_missing`
    - The values are mutated according to negabs

    The result is returned in a numpy array with the same length as the `dates` input.
    Due to the way the cubic splines are generated, there may be several dozen to a hundred data
    points at the beginning and end that are `nan`. To fix this, choose a dates array that is a
    couple of days longer than what you really want. Then, at the end, filter the output to only
    the dates in the middle.

    Parameters
    ----------
    dates: series of datetime, shape=(n_inputs,)
        The index of the time series that will be created. At least length 2.
        Must be a pandas series, with a `.dt` attribute.
    monthly: numeric, optional (default=0)
        The magnitude of the (random) monthly pattern. A random magnitude will be created for each
        month, with a cubic spline interpolating between months. The higher this value, the
        stronger the monthly pattern
    daily: numeric, optional (default=0)
        The magnitude of the (random) daily pattern. A random magnitude will be created for each
        day of the week, with a cubic spline interpolating between days. The higher this value,
        the stronger the daily pattern
    hourly: numeric, optional (default=0)
        The magnitude of the (random) hourly pattern. A random magnitude will be created for each
        hour, with a cubic spline interpolating between days. The higher this value, the stronger
        the daily pattern
    monthnoise: tuple of (str, numeric), optional (default=(None, 0))
        The type and magnitude of the monthly noise. For each month, a different magnitude will be
        uniformly drawn between 0 and `monthnoise[1]`. The type of the noise is given in
        `monthnoise[0]` and is either 'normal', 'poisson', or other (no noise). This noise is added
        to all points,but the magnitude wil differ between the 12 different months.
    daynoise: tuple of (str, numeric), optional (default=(None, 0))
        The type and magnitude of the daily noise. For each day of the week, a different magnitude
        will be drawn between 0 and `daynoise[1]`. The type of the noise is given in `daynoise[0]`
        and is either 'normal', 'poisson', or other (no noise). This noise is added to all points,
        but the magnitude wil differ between the 7 different days of the week.
    noise: dict, optional (default={})
        The types of noise that are added to every single point. The keys of this dictionary are
        'normal', 'poisson', or other (ignored)
        The value of the dictionary is the scale of the noise, standard deviation for normal noise,
        and the lambda value for poisson noise. The greater, the higher the variance of the result.
    minmax_values: tuple, optional (default=None)
        The values will be linearly rescaled to always fall within these bounds.
        By default, no rescaling is done.
    cutoff_values: tuple, optional (default=None)
        After rescaling, all the values that fall outside of these bounds will be set to `nan`.
        By default, no cutoff is done, and no values will be set to `nan`.
    negabs: numeric, optional (default=None)
        This value is subtracted from all the output (after rescaling), and then the result will
        be the absolute value. This oddly-specific operation is useful in case you want a positive
        value that has a lot of values around 0. This is very hard to do otherwise.
        By subtracting and taking the absolute value, this is achieved.
    random_missing: numeric, optional (default=None)
        Between 0 and 1. The fraction of values that will be set to nan. Used to emulate time
        series with a lot of missing values. The missing values will be completely randomly
        distributed with no pattern.
    seed: int or 1-d array_like, optional (default=None)
        seed for random noise generation. Passed through to `numpy.random.seed`. By default, no
        call to `numpy.random.seed` is made.

    Returns
    -------
    timeseries: numpy array, shape=(n_inputs,)
        A numpy array containing numbers, generated according to the provided parameters.

    Examples
    --------
    >>> # Create data that slightly resembles the temperature in a Nereda reactor:
    >>> from sam.data_sources.synthetic_data import synthetic_date_range, synthetic_timeseries
    >>> dates = pd.date_range('2015-01-01', '2016-01-01', freq='6H').to_series()
    >>> rnd = synthetic_timeseries(
    ...     dates,
    ...     monthly=5,
    ...     daily=1,
    ...     hourly=0.0,
    ...     monthnoise=('normal', 0.01),
    ...     daynoise=('normal', 0.01),
    ...     noise={'normal': 0.1},
    ...     minmax_values=(5, 25),
    ...     cutoff_values=None,
    ...     random_missing=0.12,
    ...     seed = 0,
    ... )
    >>> # visualize the result to see if it looks random or not
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax = ax.plot(dates[600:700], rnd[600:700])
    >>> fig = fig.autofmt_xdate()
    >>> plt.show()  # doctest: +SKIP
    """
    if dates.size < 2:
        raise ValueError("There must be at least 2 datetimes to generate a timeseries")
    if seed is not None:
        np.random.seed(seed)
    data = np.zeros(dates.size)

    # dt.month and dt.day start at 1, the rest start at 0
    # For this reason, 1 is subtracted from month and day, because _interpolate_pattern
    # Expects its inputs to start at 1

    # add monthly pattern
    data += _interpolate_pattern(dates.dt.month - 1, dates.dt.day - 1, monthly, 12)
    # add daily pattern
    data += _interpolate_pattern(dates.dt.dayofweek, dates.dt.hour, daily, 7)
    # add hourly pattern
    data += _interpolate_pattern(dates.dt.hour, None, hourly, 24)

    # add monthly noise
    data += _add_temporal_noise(dates.dt.month - 1, monthnoise[0], monthnoise[1], 12)
    # add monthly noise
    data += _add_temporal_noise(dates.dt.dayofweek, daynoise[0], daynoise[1], 7)

    # add noise
    for key in noise:
        if key == "normal":
            data += np.random.normal(size=dates.size, loc=0, scale=noise["normal"])
        if key == "poisson":
            data += np.random.poisson(size=dates.size, lam=noise["poisson"])

    # Rescale values to all fall exactly in the minmax values
    if minmax_values is not None:
        currentmin, currentmax = np.nanmin(data), np.nanmax(data)
        scale = (minmax_values[1] - minmax_values[0]) / (currentmax - currentmin)
        offset = minmax_values[0] - currentmin
        data = scale * (data + offset)

    # Set values outside of cutoff to nan. Used for more predictable nans
    if cutoff_values is not None:
        data = np.where((data < cutoff_values[0]) | (data > cutoff_values[1]), np.nan, data)

    # Subtracts a value from the result, and absolute value.
    # This makes the result center more around 0
    if negabs is not None:
        data = np.abs(data - negabs)

    # Add random missing values, to the proportion of random_missing.
    if random_missing is not None:
        data[np.random.choice(data.size, int(data.size * random_missing), replace=False)] = np.nan

    return data


def synthetic_date_range(
    start="2016-01-01",
    end="2017-01-01",
    freq="h",
    max_delay=0,
    random_stop_freq=0,
    random_stop_max_length=1,
    seed=None,
):
    """
    Create a synthetic, somewhat realistic-looking array of datetimes.

    Given a start time, end time, frequency, and some variables governing noise,
    creates an array of datetimes that is somewhat random.

    The algorithm:

    - Generate a regular pandas date_range with start, end, and frequency
    - Delay each time by a uniformly chosen random number between 0 and `max_delay`, in seconds.
    - Pick a proportion `random_stop_freq` of times randomly. Each of these times `x_i`
      are deemed 'stoppages', and for each, a number between 1 and `random_stop_max_length` is
      uniformly chosen, say `k_i`. Then, the 'stoppage', the `k_i` next points after `x_i` are
      deleted, causing a hole in the times.
    - Only the times strictly smaller than end are kept. This means end is an exclusive bound.

    Parameters
    ----------
    start: str or datetime-like, optional (default='2016-01-01')
        Left bound for generating dates.
    end: str or datetime-like, optional (default='2017-01-01')
        Right bound for generating dates. Exclusive bound.
    freq: str or DateOffset, optional (default='h') (hourly)
        Frequency strings can have multiples, e.g. '5H'. See `here for a list of frequency aliases.
        <https://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases`_
    max_delay: numeric, optional (default=0)
        Each time is delayed by a random number of seconds, chosen between 0 and `max_delay`
    random_stop_freq: numeric, optional (default=0)
        Number between 0 and 1. This proportion of all times are deemed as starting points of
        'stoppages'. A stoppage means that a number of points are removed from the result.
    random_stop_max_length: numeric, optional (default=1)
        Each stoppage will have a randomly generated length, between 1 and
        `random_stop_max_length`. A stoppage of length `k` means that the first `k` points after
        the start of the stoppage are deleted.
    seed: int or 1-d array_like, optional (default=None)
        seed for random noise generation. Passed through to `numpy.random.seed`. By default, no
        call to `numpy.random.seed` is made.

    Returns
    -------
    rng: DatetimeIndex
        A pandas datetimeindex of noisy times

    Examples
    --------
    >>> # Generate times with point approximately every 6 hours
    >>> from sam.data_sources.synthetic_data import synthetic_date_range
    >>> synthetic_date_range('2016-01-01', '2016-01-02', '6h', 600, 0, 1, seed=0)
    DatetimeIndex(['2016-01-01 00:05:29.288102356',
                   '2016-01-01 06:12:38.401722180',
                   '2016-01-01 12:18:40.059747823',
                   '2016-01-01 18:24:06.989657621'],
                  dtype='datetime64[ns]', freq=None)

    >>> # Generate times with very likely stops of length 1
    >>> synthetic_date_range('2016-01-01', '2016-01-02', 'h', 0, 0.5, 1, seed=0)
    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00',
                   '2016-01-01 02:00:00', '2016-01-01 03:00:00',
                   '2016-01-01 04:00:00', '2016-01-01 05:00:00',
                   '2016-01-01 09:00:00', '2016-01-01 10:00:00',
                   '2016-01-01 11:00:00', '2016-01-01 13:00:00',
                   '2016-01-01 16:00:00', '2016-01-01 21:00:00'],
                  dtype='datetime64[ns]', freq=None)
    """
    index = pd.date_range(start, end, freq=freq).to_series()
    if index.size == 0:
        raise ValueError("End time must be after start time.")
    if seed is not None:
        np.random.seed(seed)

    # Add a delay of 0 to max_delay seconds to every time
    random_delay = np.cumsum(np.random.uniform(0, max_delay, index.size))
    index += pd.to_timedelta(random_delay, unit="s")

    # Choose indexes for the start of random stops
    random_stops = np.random.choice(len(index), int(random_stop_freq * len(index)), replace=False)
    for ix in random_stops:
        # If random_stop_max_length is 1, we cannot randomly draw a number, so just set it to 1
        stop_length = (
            1 if random_stop_max_length == 1 else np.random.randint(1, random_stop_max_length)
        )
        index[ix : ix + stop_length] = np.nan
    # Remove the stops. This causes long batches that take (much) longer than usual.
    index = index.dropna()
    # make the end an exclusive bound. It may have been surpassed because of the delays
    index = index[index < pd.to_datetime(end)]

    return pd.DatetimeIndex(index)
