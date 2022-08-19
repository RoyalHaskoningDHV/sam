from typing import List, Union

import numpy as np
import pandas as pd


def compute_quantile_ratios(
    y: pd.Series, pred: pd.DataFrame, predict_ahead: int = 0, precision: int = 3
):
    """
    Computes the total proportion of data points in y beneath each quantile in pred.
    So for example, with quantile 0.75, you'd expect 75% of values in y to be beneath this quantile
    This function would then return a dictionary with 0.75 as key, and the actual proportion of
    values that fell beneath that quantile in y (e.g. 0.743).
    This allows the user to judge whether the fitted quantile model accurately reflects the
    distribution of the data.

    Parameters
    ---------
    y: pd.Series
        series of observed true values
    pred: pd.DataFrame
        output of MLPTimeseriesRegressor.predict() function
    predict_ahead: int (default=0)
        Number of timestep ahead to evaluate
    precision: int (default=3)
        decimal point precision of result

    Returns
    -------
    quantile_ratios: dict
        key is quantile, value is the observed proportion of data points below that quantile
    """

    # derive quantiles from column names
    qs = [float(c.split("_")[-1]) for c in pred.columns if "mean" not in c]

    quantile_ratios = {
        # mean here computes ratio (mean of True/False - 0/1s)
        q: (y < pred["predict_lead_%d_q_" % predict_ahead + str(q)]).mean()
        for q in qs
    }

    quantile_ratios = {
        round(key, precision): round(value, precision) for key, value in quantile_ratios.items()
    }

    return quantile_ratios


def compute_quantile_crossings(
    pred: pd.DataFrame, predict_ahead: int = 0, qs: List[Union[float, str]] = None
):
    """
    Computes the total proportion of predictions of a certain quantile that fall below the next
    lower quantile. This phenomenon is called 'quantile crossing'.
    So for example, with quantiles [0.1, 0.25, 0.75, 0.9], you'd expect all predictions of
    the 0.9 quantile to be above the 0.75 quantile predictions.
    In this example, this function would calculate the proportion of observations where the 0.9
    prediction actually falls below the 0.75 prediction, the 0.75 below the 0.25, and the 0.25
    below the 0.1 predictions.
    In addition, this function calculates in what proportion of cases the mean prediction falls
    outside of the closest quantile border below and above 0.5.
    NOTE: this function operates on the output of MLPTimeseriesRegressor.predict()
    NOTE-2: this function expects that either 0.5 or 'mean' is in the qs, or in the columns of pred
    when qs=None.

    Parameters
    ---------
    pred: pd.DataFrame
        Output of MLPTimeseriesRegressor.predict() function
    predict_ahead: int (default=0)
        Number of timestep ahead to evaluate
    qs: list (default=None)
        List of quantiles. If none, uses all quantiles in the pred columns.
        qs can be provided to compare crossings between a specific subset of quantiles.
        You can also add 'mean' to this list to add it to the comparison. It will then
        be compared to the nearest quantiles above and below 0.5.

    Returns
    -------
    crossings: dict
        Key is quantile, value is the proportion of quantile crossings.
    """

    # switch mean with 0.5 for ease in rest of function
    if qs is None:
        qs = [float(c.split("_")[-1]) for c in pred.columns if "mean" not in c] + [0.5]
        if qs.count(0.5) > 1:
            raise ValueError("0.5 and 'mean' cannot both be in qs")
    else:
        if 0.5 in qs and "mean" in qs:
            raise ValueError("0.5 and 'mean' cannot both be in qs")
        qs = [0.5 if q == "mean" else q for q in qs]

    # now replace the 'mean' part with 0.5 in the predictions
    pred.columns = [c.replace("mean", "q_0.5") for c in pred.columns]

    # make sure quantiles are sorted if they aren't already:
    qs = np.sort(qs)[::-1]

    # now compute the quantile crossings
    crossings = {}
    for c in range(len(qs) - 1):
        crossings[f"{qs[c]:.3f} < {qs[c+1]:.3f}"] = (
            pred[f"predict_lead_{predict_ahead}_q_{qs[c]}"]
            < pred[f"predict_lead_{predict_ahead}_q_{qs[c+1]}"]
        ).mean()

    # now replace 0.5 with mean again
    crossings = {key.replace("0.500", "mean"): value for key, value in crossings.items()}

    return crossings
