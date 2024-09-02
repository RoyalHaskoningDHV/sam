import logging

import numpy as np
import pandas as pd
from sam.logging_functions import log_dataframe_characteristics

logger = logging.getLogger(__name__)


def incident_curves(
    data: pd.DataFrame,
    under_conf_interval: bool = True,
    max_gap: int = 0,
    min_duration: int = 0,
    max_gap_perc: float = 1,
    min_dist_total: float = 0,
    actual: str = "ACTUAL",
    low: str = "PREDICT_LOW",
    high: str = "PREDICT_HIGH",
):
    """
    Finds and labels connected outliers, or 'curves'. The basic idea of this function is to define
    an outlier as a row where the value is outside some interval. 'Interval' here refers to a
    prediction interval, that can be different for every row. This interval defines what a model
    considers a 'normal' value for that datapoint. Missing values are not considered outliers.

    Then, we apply various checks and filters to the outlier to create 'curves', or streaks of
    connected outlies. These curves can have gaps, if `max_gap > 0`. In the end, only the curves
    that satisfy conditions are kept. Curves that do not satisfy one of the conditions are ignored
    (essentially, the output will act as if they are not outliers at all).

    The output is an array of the same length as the number of rows as data, with each streak of
    outliers labeled with an unique number.
    This algorithm assumes the input is sorted by time, adjacent rows are adjacent measurements!

    Parameters
    ----------
    data: pd.DataFrame (n_rows, _)
        dataframe containing values, as well two columns determining an interval. These three
        column names can be configured using the 'actual', 'low', and 'high' parameters
    under_conf_interval: bool, optional (default=True)
        If true, values lower than the interval count as outliers. Else, only
        values higher than the interval are counted as outliers.
    max_gap: int, optional (default=0)
        How many gaps are allowed between outliers. For example, if max_gap = 2, and the outliers
        look like: [True, False, False, True], then this 'gap' of 2 will be assigned to this
        curve, and this will turn into a single curve of outliers, of length 4
    min_duration: int, optional (default=0)
        Minimum number of outliers per curve. Curves with a smaller length than this value,
        will be ignored. Gaps are counted in the duration of a curve.
    max_gap_perc: float, optional (default=1)
        The maximum percentage of gaps that a curve can have to count. For example,
        if this is 0.4, and a curve contains 2 outliers and 2 gaps, then the curve will be
        ignored.
    min_dist_total: float, optional (default=0)
        If a curve should have a minimum 'outlier size', and if so, how much.
        The outlier size here is defined as the distance between the value and the end of
        the interval. For example, if the interval is (10, 20) and the value is 21, the
        'outlier size' is 1. These values are summed (gaps are counted as 0), and compared
        to this value. Curves with a sum that is too low will be ignored.
    actual: string, optional (default='ACTUAL')
        The name of the column in the data containing the value for each row
    low: string, optional (default='PREDICT_LOW')
        The name of the column in the data containing the lower end of the interval for each row
    high: string, optional (default='PREDICT_HIGH')
        The name of the column in the data containing the higher end of the interval for each row

    Returns
    -------
    outlier_curves: array-like (n_rows,)
        A numpy array of numbers labeling each curve. 0 means there is no outlier curve
        here that satisfies all conditions.

    Examples
    --------
    >>> from sam.exploration import incident_curves
    >>> data = pd.DataFrame({'ACTUAL': [0.3, np.nan, 0.3, np.nan, 0.3, 0.5, np.nan, 0.7],
    ...                      'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4})
    >>> incident_curves(data)
    array([1, 0, 2, 0, 3, 0, 0, 4])
    >>> incident_curves(data, max_gap=1)
    array([1, 1, 1, 1, 1, 0, 0, 2])
    >>> incident_curves(data, max_gap=1, max_gap_perc=0.2)
    array([0, 0, 0, 0, 0, 0, 0, 2])
    """

    def _number_true_streaks(series):
        """Given a boolean series, numbers series of True and set False to 0. For example,
        [T, T, F, T, F, T, T] will be numbered as [1, 1, 0, 2, 0, 3, 3]. Each streak of T gets
        the same number. The result will be converted to numpy array.
        """
        begin_streak = series.ne(series.shift())
        # We only want beginning of True streaks, not False
        begin_true_streak = series & begin_streak
        return np.where(series, begin_true_streak.cumsum(), 0)

    logger.debug(
        "Finding outlier curves: under_conf_interval={}, max_gap={}, min_duration={}, "
        "max_gap_perc={}, min_dist_total={}, actual={}, low={}, high={}".format(
            under_conf_interval,
            max_gap,
            min_duration,
            max_gap_perc,
            min_dist_total,
            actual,
            low,
            high,
        )
    )

    data = data.copy()  # copy because we are going to mess with this df in various ways
    data = data.rename(columns={actual: "ACTUAL", low: "PREDICT_LOW", high: "PREDICT_HIGH"})
    # Three columns needed: ACTUAL, PREDICT_HIGH, PREDICT_LOW
    data["OUTLIER"] = (
        (data.ACTUAL > data.PREDICT_HIGH) | (data.ACTUAL < data.PREDICT_LOW)
        if under_conf_interval
        else (data.ACTUAL > data.PREDICT_HIGH)
    )
    # Find the streaks of gaps. Gaps are defined as anything that's not an outlier
    data["GAP"] = _number_true_streaks(~data["OUTLIER"])
    # Then, all gaps with length of max_gap or lower are merged with neighbouring outliers
    new_val = data.groupby("GAP").apply(lambda x: True if (x.shape[0] <= max_gap) else False)
    new_val[0] = True  # because this is not an outlier, it's treated seperately
    new_val.name = "OUTLIER_FILLED"  # Attribute needed for join
    data = data.join(new_val, on="GAP")  # Add OUTLIER_FILLED column

    # If there is an outlier AND a gap at the beginning/end. That is wrong.
    firstgap_id, lastgap_id = data.GAP.iloc[0], data.GAP.iloc[-1]
    if firstgap_id != 0:  # entire gap at the beginning that shouldn't be there
        data.loc[data.GAP == firstgap_id, "OUTLIER_FILLED"] = False
    if lastgap_id != 0:  # entire gap at the end that shouldn't be there
        data.loc[data.GAP == lastgap_id, "OUTLIER_FILLED"] = False

    # Calculate OUTLIER_DIST which is needed to interpret min_dist_total parameter
    data["OUTLIER_DIST"] = np.where(
        data.OUTLIER,
        np.where(
            data.ACTUAL > data.PREDICT_HIGH,
            data.ACTUAL - data.PREDICT_HIGH,
            data.PREDICT_LOW - data.ACTUAL if under_conf_interval else 0,
        ),
        0,
    )

    # Now calculate the streaks of outliers, and number them
    data["OUTLIER_CURVE"] = _number_true_streaks(data["OUTLIER_FILLED"])

    logging.debug("Curves found after fixing gaps: {}".format(data["OUTLIER_CURVE"].max()))

    # Lastly, filter the outlier streaks that don't match one of the three criteria from params.
    real_outlier = data.groupby("OUTLIER_CURVE").apply(
        lambda x: (x.shape[0] >= min_duration)
        and (x.OUTLIER_DIST.sum() >= min_dist_total)
        and (1 - (x.OUTLIER.sum() / x.shape[0]) <= max_gap_perc)
    )
    real_outlier.name = "REAL_OUTLIER"
    data = data.join(real_outlier, on="OUTLIER_CURVE")

    logging.info("Curves found in final result: {}".format(data["OUTLIER_CURVE"].max()))
    return np.where(data.REAL_OUTLIER, data.OUTLIER_CURVE, 0)


def incident_curves_information(
    data: pd.DataFrame,
    under_conf_interval: bool = True,
    return_aggregated: bool = True,
    normal: str = "PREDICT",
    time: str = "TIME",
    **kwargs,
):
    """
    Aggregates a dataframe by incident_curves.
    This function calculates incident_curves using incident_curves, and then calculates
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
    data: pd.DataFrame (n_rows, _)
        dataframe containing actual, low, high, normal, and time columns. These column names
        can be configured using those parameters. The default values are (ACTUAL, PREDICT_LOW,
        PREDICT_HIGH, PREDICT, TIME)
    under_conf_interval: bool, optional (default=True)
        If true, values lower than the interval count as outliers. Else, only
        values higher than the interval are counted as outliers.
    return_aggregated: bool, optional (default=True)
        If true the information about the outliers will be aggregated by OUTLIER_CURVE.
        Else, information will not be aggregated. The two options return different
        types of information.
    normal: string, optional (default='PREDICT')
        The name of the column in the data containing a 'normal' value for each row
    time: string, optional (default='TIME')
        The name of the column in the data containing a 'time' value for each row

    Returns
    -------
    information, dataframe
        if return_aggregated is false: information about each outlier.
        The output will have the folowing columns:

        - all the original columns
        - OUTLIER (bool) whether the value of the row is considered an outlier
        - OUTLIER_CURVE (numeric) the streak the outlier belongs to, or 0 if it's
          not an outlier
        - OUTLIER_DIST (numeric) The distance between the value and the outside of
          the interval, describing how much 'out of the normal' the value is
        - OUTLIER_SCORE (numeric) If x is OUTLIER_DIST, and y is the distance
          between the value and the 'normal' column, then this is x / (1 + y)
          This defines some 'ratio' of how abnormal the value is. This can be useful
          in scale-free data, where the absolute distance is not a fair metric.
        - OUTLIER_TYPE (string) 'positive' if the outlier is above the interval,
          and 'negative' if the outlier is below the interval

        if `return_aggregated`, then it will return information about each outlier curve
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
    >>> data = pd.DataFrame({'TIME': range(1547477436, 1547477436+3),  # unix timestamps
    ...                     'ACTUAL': [0.3, 0.5, 0.7],
    ...                     'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4, 'PREDICT': 0.5})
    >>> incident_curves_information(data)  # doctest: +ELLIPSIS
                   OUTLIER_DURATION  ...
    >>> incident_curves_information(data, return_aggregated=False)
             TIME  ACTUAL  PREDICT_HIGH  ...  OUTLIER_DIST  OUTLIER_SCORE  OUTLIER_TYPE
    0  1547477436     0.3           0.6  ...           0.1       0.090909      negative
    1  1547477437     0.5           0.6  ...           0.0       0.000000          none
    2  1547477438     0.7           0.6  ...           0.1       0.090909      positive
    <BLANKLINE>
    [3 rows x 10 columns]
    """
    data = data.copy()
    data = data.rename(columns={normal: "PREDICT", time: "TIME"})
    logging.debug("Creating outlier information: return_aggregated={}".format(return_aggregated))
    data["OUTLIER_CURVE"] = incident_curves(data, under_conf_interval, **kwargs).astype(np.int64)
    # On unix, 64 is already the default, but on windows, the
    # outlier_curves function returns 32 bit integers. This function returns a numpy array
    # which is consistent across platforms, so values are converted to 64-bit to ensure
    # consistency.

    data["OUTLIER"] = (
        (data.ACTUAL > data.PREDICT_HIGH) | (data.ACTUAL < data.PREDICT_LOW)
        if under_conf_interval
        else (data.ACTUAL > data.PREDICT_HIGH)
    )
    data["OUTLIER_DIST"] = np.where(
        data.OUTLIER,
        np.where(
            data.ACTUAL > data.PREDICT_HIGH,
            data.ACTUAL - data.PREDICT_HIGH,
            data.PREDICT_LOW - data.ACTUAL if under_conf_interval else 0,
        ),
        0,
    )
    data["OUTLIER_SCORE"] = np.where(
        data.OUTLIER,
        np.where(
            data.ACTUAL > data.PREDICT_HIGH,
            (data.ACTUAL - data.PREDICT_HIGH) / (1 + data.PREDICT_HIGH - data.PREDICT),
            (
                (data.PREDICT_LOW - data.ACTUAL) / (1 + data.PREDICT - data.PREDICT_LOW)
                if under_conf_interval
                else 0
            ),
        ),
        0,
    )
    data["OUTLIER_TYPE"] = np.where(
        data.ACTUAL > data.PREDICT_HIGH,
        "positive",
        np.where(data.ACTUAL < data.PREDICT_LOW, "negative", "none"),
    )

    if not return_aggregated:
        return data

    streaks = data.groupby("OUTLIER_CURVE").agg(
        {
            "OUTLIER_CURVE": "count",
            "OUTLIER_DIST": ["sum", "max"],
            "OUTLIER_SCORE": ["max"],
            "TIME": ["min", "max"],
            "OUTLIER_TYPE": lambda x: x.iloc[0],
        }
    )
    streaks.columns = ["_".join(x) for x in streaks.columns]
    streaks = streaks.rename(
        columns={
            "OUTLIER_CURVE_count": "OUTLIER_DURATION",
            "TIME_min": "OUTLIER_START_TIME",
            "TIME_max": "OUTLIER_END_TIME",
            "OUTLIER_SCORE_max": "OUTLIER_SCORE_MAX",
            "OUTLIER_DIST_sum": "OUTLIER_DIST_SUM",
            "OUTLIER_DIST_max": "OUTLIER_DIST_MAX",
            "OUTLIER_TYPE_<lambda>": "OUTLIER_TYPE",
        }
    )
    # reorder columns
    streaks = streaks[
        [
            "OUTLIER_DURATION",
            "OUTLIER_TYPE",
            "OUTLIER_SCORE_MAX",
            "OUTLIER_START_TIME",
            "OUTLIER_END_TIME",
            "OUTLIER_DIST_SUM",
            "OUTLIER_DIST_MAX",
        ]
    ]
    logger.info("Created data from incident_curves_information:")
    log_dataframe_characteristics(streaks, logging.INFO)
    return streaks[streaks.index != 0]
