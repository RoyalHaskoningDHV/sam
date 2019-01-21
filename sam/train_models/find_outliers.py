import pandas as pd
import numpy as np


def find_outlier_curves(data, under_conf_interval=True, max_gap=0, min_duration=0, max_gap_perc=1,
                        min_dist_total=0, actual='ACTUAL', low='PREDICT_LOW', high='PREDICT_HIGH'):
    """
    Finds and labels connected outliers, or 'curves'. The basic idea of this function is to define
    an outlier as a row where the value is outside some interval. 'Interval' here refers to a
    prediction interval, that can be different for every row. This interval defines what a model
    considers a 'normal' value for that timepoint. Missing values are never an
    outlier. Then, we apply various checks and filters to it to create 'curves', or streaks of
    connected outlies. These curves can have gaps, if max_gap > 0. In the end, only the curves that
    satisfy conditions are kept. Curves that do not satisfy one of the conditions are ignored
    (essentially, the output will act as if they are not outliers at all).
    The output is an array of the same length as the number of rows as data, with each streak of
    outliers labeled with an unique number.
    This algorithm assumes the input is sorted by time, adjacent rows are adjacent measurements!

    Parameters
    ----------
    data: dataframe (n_rows, _)
        dataframe containing values, as well two columns determining an interval. These three
        column names can be configured using the 'actual', 'low', and 'high' parameters
    under_conf_interval: boolean, optional (default=True)
        If values lower than the interval count as outliers. If this is false, then only
        values higher than the interval are counted as outliers. false by default
    max_gap: numeric, optional (default=0)
        If gaps should be allowed in the outliers, and if yes, how long they can be.
        For example, if max_gap = 2, and the outliers look like: [True, False, False, True],
        then this 'gap' of 2 will be assigned to this curve, and this will turn into a single
        curve of outliers, of length 4
    min_duration: numeric, optional (default=0)
        If outliers should have a minimum duration. If yes, how long. Curves with a length
        that is smaller than this value, will be ignored. Gaps are counted in the duration
        of a curve.
    max_gap_perc: numeric, optional (default=1)
        The maximum percentage of gaps that a curve can have to count. For example,
        if this is 0.4, and a curve contains 2 outliers and 2 gaps, then the curve will be
        ignored.
    min_dist_total: numeric, optional (default=0)
        If a curve should have a minimum 'outlier size', and if so, how much.
        The outlier size here is defined as the distance between the value and the end of
        the interval. For example, if the interval is (10, 20) and the value is 21, the
        'outlier size' is 1. These values are summed (gaps are counted as 0), and compared
        to this value. Curves with a sum that is too low will be ignored.
    actual: string, optional (default='ACTUAL')
        The name of the colomn in the data containing the value for each row
    low: string, optional (default='PREDICT_LOW')
        The name of the colomn in the data containing the lower end of the interval for each row
    high: string, optional (default='PREDICT_HIGH')
        The name of the colomn in the data containing the higher end of the interval for each row

    Returns
    -------
    outlier_curves: array-like (n_rows,)
        A numpy array of numbers labeling each curve. 0 means there is no outlier curve
        here that satisfies all conditions.

    Examples
    --------
    >>> from sam.train_model import find_outlier_curve
    >>> data = pd.DataFrame({'ACTUAL': [0.3, np.nan, 0.3, np.nan, 0.3, 0.5, np.nan, 0.7],
    >>>                      'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4})
    >>> find_outlier_curve(data)
    array([1, 0, 2, 0, 3, 0, 0, 4])
    >>> find_outlier_curve(data, max_gap=1)
    array([1, 1, 1, 1, 1, 0, 0, 2])
    >>> find_outlier_curve(data, max_gap=1, max_gap_perc=0.2)
    array([0, 0, 0, 0, 0, 0, 0, 2])
    """
    def _firstfalse(series):
        # Helper function
        series[0] = False
        return series

    data = data.copy()  # copy because we are going to mess with this df in various ways
    data = data.rename(columns={actual: 'ACTUAL', low: 'PREDICT_LOW', high: 'PREDICT_HIGH'})
    # Three columns needed: ACTUAL, PREDICT_HIGH, PREDICT_LOW
    data['OUTLIER'] = (data.ACTUAL > data.PREDICT_HIGH) | (data.ACTUAL < data.PREDICT_LOW) \
        if under_conf_interval else (data.ACTUAL > data.PREDICT_HIGH)
    # First of all, find all the streaks of gaps, and number then. Gaps are opposite of outliers
    # shift sets the first value to NaN, so we have to set it to False manually.
    data['GAP'] = np.where(data.OUTLIER, 0,
                           (~data.OUTLIER - _firstfalse((~data.OUTLIER).shift()) == 1).cumsum())
    # Then, all gaps with length of max_gap or lower are merged with neighbouring outliers
    new_val = data.groupby('GAP').apply(lambda x: True if (x.shape[0] <= max_gap) else False)
    new_val[0] = True  # because this is not an outlier, it's treated seperately
    new_val.name = 'OUTLIER_FILLED'  # Attribute needed for join
    data = data.join(new_val, on='GAP')  # Add OUTLIER_FILLED column

    # If there is an outlier AND a gap at the beginning/end. That is wrong.
    firstgap_id, lastgap_id = data.GAP.iloc[0], data.GAP.iloc[-1]
    if firstgap_id != 0:  # entire gap at the beginning that shouldn't be there
        data.loc[data.GAP == firstgap_id, 'OUTLIER_FILLED'] = False
    if lastgap_id != 0:  # entire gap at the end that shouldn't be there
        data.loc[data.GAP == lastgap_id, 'OUTLIER_FILLED'] = False

    # Calculate OUTLIER_DIST which is needed to interpret min_dist_total parameter
    data['OUTLIER_DIST'] = np.where(data.OUTLIER, np.where(data.ACTUAL > data.PREDICT_HIGH,
                                                           data.ACTUAL - data.PREDICT_HIGH,
                                                           data.PREDICT_LOW - data.ACTUAL
                                                           if under_conf_interval else 0), 0)

    # Now calculate the streaks of outliers, and number them
    data['OUTLIER_CURVE'] = np.where(
        data.OUTLIER_FILLED,
        (data.OUTLIER_FILLED - _firstfalse(data.OUTLIER_FILLED.shift()) == 1).cumsum(), 0)

    # Lastly, filter the outlier streaks that don't match one of the three criteria from params.
    real_outlier = data.groupby('OUTLIER_CURVE').apply(
        lambda x: (x.shape[0] >= min_duration) and
                  (x.OUTLIER_DIST.sum() >= min_dist_total) and
                  (1 - (x.OUTLIER.sum() / x.shape[0]) <= max_gap_perc))
    real_outlier.name = 'REAL_OUTLIER'
    data = data.join(real_outlier, on='OUTLIER_CURVE')
    return np.where(data.REAL_OUTLIER, data.OUTLIER_CURVE, 0)


def create_outlier_information(data, under_conf_interval=True, return_aggregated=True,
                               normal='PREDICT', time='TIME', **kwargs):
    """
    Aggregates a dataframe by outlier curves.
    This function calculates outlier curves using find_outlier_curve, and then calculates
    information about each outlier curve. This function can either return raw information
    about each outlier row, keeping the number of rows the same, or it can aggregate the dataframe
    by outlier curve, which means there will be specific information per curve, such as the length,
    the total outlier distance, etcetera.

    The data must contain the actual, low, high values (actual is the actual value, and low/high
    are some interval deeming what is 'normal') Also, the data must contain a 'time' column.
    Lastly, the data must contain a 'normal' column, describing what would be considered
    the most normal value (for example the middle of the interval)
    This algorithm assumes the input is sorted by time, adjacent rows are adjacent measurements!

    Parameters
    ----------
    data: dataframe (n_rows, _)
        dataframe containing actual, low, high, normal, and time columns. These column names
        can be configured using those parameters. The default values are (ACTUAL, PREDICT_LOW,
        PREDICT_HIGH, PREDICT, TIME)
    under_conf_interval: boolean, optional (default=True)
        If values lower than the interval count as outliers. If this is false, then only
        values higher than the interval are counted as outliers. false by default
    return_aggregated: boolean, optional (default=True)
        Wether the information about the outliers should be aggregated by outlier_curve.
        The two options return different types of information.
    normal: string, optional (default='PREDICT')
        The name of the colomn in the data containing a 'normal' value for each row
    time: string, optional (default='TIME')
        The name of the colomn in the data containing a 'time' value for each row

    Returns
    -------
    information, dataframe
        if return_aggregated is false: information about each outlier.
        The output will have the folowing columns:
        - all the original columns
        - OUTLIER (boolean) whether the value of the row is considered an outlier
        - OUTLIER_CURVE (numeric) the streak the outlier belongs to, or 0 if it's
          not an outlier
        - OUTLIER_DIST (numeric) The distance between the value and the outside of
          the interval, describing how much 'out of the normal' the vallue is
        - OUTLIER_SCORE (numeric) If x is OUTLIER_DIST, and y is the distance
          between the value and the 'normal' column, then this is x / (1 + y)
          This defines some 'ratio' of how abnormal the value is. This can be useful
          in scale-free data, where the absolute distance is not a fair metric.
        - OUTLIER_TYPE (string) 'positive' if the outlier is above the interval,
          and 'negative' if the outlier is below the interval

        if return_aggregated, then it will return information about each outlier curve
        The output will have the following columns:
        - index: OUTLIER_CURVE (numeric) The id of the curve. 0 is not included
        - OUTLIER_DURATION (numeric) The number of points in the curve, including gaps
        - OUTLIER_TYPE (string) if the first point is positive or negative. Other points
          in the curve may have other types
        - OUTLIER_SCORE_MAX (numeric) The maximum of OUTLIER_SCORE of all the points
          in the curve
        - OUTLIER_START_TIME (datetime) The value of the 'time' column of the first point
          in the curve
        - OUTLIER_END_TIME (datetime) The value of the 'time' column of the last point
          in the curve
        - OUTLIER_DIST_SUM (numeric) The sum of OUTLIER_DIST of the points in the curve.
          Gaps count as 0
        - OUTLIER_DIST_MAX (numeric) The max of OUTLIER_DIST of the points in the curve

    Examples
    --------
    >>>data = pd.DataFrame({'TIME': range(1547477436, 1547477436+8),  # unix timestamps
    >>>                    'ACTUAL': [0.3, 0.5, 0.7],
    >>>                    'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4, 'PREDICT': 0.5})
    >>>create_outlier_information(data)
                       OUTLIER_DURATION  OUTLIER_START_TIME  OUTLIER_END_TIME  \
    OUTLIER_CURVE
    1                             1          1547477436        1547477436
    2                             1          1547477438        1547477438

                   OUTLIER_SCORE_MAX  OUTLIER_DIST_SUM  OUTLIER_DIST_MAX  \
    OUTLIER_CURVE
    1                       0.090909               0.1               0.1
    2                       0.090909               0.1               0.1

                  OUTLIER_TYPE
    OUTLIER_CURVE
    1                 negative
    2                 positive

    >>>create_outlier_information(data, return_aggregated=False)
           ACTUAL  PREDICT  PREDICT_HIGH  PREDICT_LOW        TIME  OUTLIER_CURVE  \
    0     0.3      0.5           0.6          0.4  1547477436              1
    1     0.5      0.5           0.6          0.4  1547477437              0
    2     0.7      0.5           0.6          0.4  1547477438              2

       OUTLIER  OUTLIER_DIST  OUTLIER_SCORE OUTLIER_TYPE
    0     True           0.1       0.090909     negative
    1    False           0.0       0.000000     negative
    2     True           0.1       0.090909     positive
    """
    data = data.copy()
    data = data.rename(columns={normal: 'PREDICT', time: 'TIME'})
    data['OUTLIER_CURVE'] = find_outlier_curves(data, **kwargs)

    data['OUTLIER'] = (data.ACTUAL > data.PREDICT_HIGH) | (data.ACTUAL < data.PREDICT_LOW) \
        if under_conf_interval else (data.ACTUAL > data.PREDICT_HIGH)
    data['OUTLIER_DIST'] = np.where(data.OUTLIER, np.where(data.ACTUAL > data.PREDICT_HIGH,
                                                           data.ACTUAL - data.PREDICT_HIGH,
                                                           data.PREDICT_LOW - data.ACTUAL
                                                           if under_conf_interval else 0), 0)
    data['OUTLIER_SCORE'] = np.where(data.OUTLIER, np.where(
        data.ACTUAL > data.PREDICT_HIGH,
        (data.ACTUAL - data.PREDICT_HIGH) / (1 + data.PREDICT_HIGH - data.PREDICT),
        (data.PREDICT_LOW - data.ACTUAL) / (1 + data.PREDICT - data.PREDICT_LOW)
        if under_conf_interval else 0), 0)
    data['OUTLIER_TYPE'] = np.where(data.ACTUAL > data.PREDICT_HIGH, "positive",
                                    np.where(data.ACTUAL < data.PREDICT_LOW, "negative", "none"))

    if not return_aggregated:
        return data

    streaks = data.groupby('OUTLIER_CURVE').agg({
        'OUTLIER_CURVE': 'count',
        'OUTLIER_DIST': ['sum', 'max'],
        'OUTLIER_SCORE': ['max'],
        'TIME': ['min', 'max'],
        'OUTLIER_TYPE': lambda x: x.iloc[0]
    })
    streaks.columns = ["_".join(x) for x in streaks.columns.ravel()]
    streaks = streaks.rename(columns={
        'OUTLIER_CURVE_count': 'OUTLIER_DURATION',
        'TIME_min': 'OUTLIER_START_TIME',
        'TIME_max': 'OUTLIER_END_TIME',
        'OUTLIER_SCORE_max': 'OUTLIER_SCORE_MAX',
        'OUTLIER_DIST_sum': 'OUTLIER_DIST_SUM',
        'OUTLIER_DIST_max': 'OUTLIER_DIST_MAX',
        'OUTLIER_TYPE_<lambda>': 'OUTLIER_TYPE'
    })
    # reorder columns
    streaks = streaks[['OUTLIER_DURATION',
                       'OUTLIER_TYPE', 'OUTLIER_SCORE_MAX',
                       'OUTLIER_START_TIME', 'OUTLIER_END_TIME',
                       'OUTLIER_DIST_SUM', 'OUTLIER_DIST_MAX']]
    return streaks[streaks.index != 0]
