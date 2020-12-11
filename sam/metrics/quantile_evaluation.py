import numpy as np


def compute_quantile_ratios(y, pred, predict_ahead=0):
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
        output of SamQuantileMLP.predict() function
    predict_ahead: int (default=0)
        predict ahead to evaluate

    Returns
    -------
    quantile_ratios: dict
        key is quantile, value is the observed proportion of data points below that quantile
    """

    # derive quantiles from column names
    qs = [float(c.split('_')[-1]) for c in pred.columns if 'mean' not in c]

    quantile_ratios = {
        # mean here computes ratio (mean of True/Falses - 0/1s)
        q: (y < pred['predict_lead_%d_q_' % predict_ahead + str(q)]).mean()
        for q in qs
    }

    return quantile_ratios


def compute_quantile_crossings(y, pred, predict_ahead=0):
    """
    Computes the total proportion of predictions of a certain quantile that fall below the next
    lower quantile. This phenomenon is called 'quantile crossing'.
    So for example, with quantiles [0.1, 0.25, 0.75, 0.9], you'd expect all predictions of
    the 0.9 quantile to be above the 0.75 quantile predictions.
    In this example, this function would calculate the proportion of observations where the 0.9
    prediction actually falls below the 0.75 prediction, the 0.75 below the 0.25, and the 0.25
    below the 0.1 predictions.
    In addition, this function calculates in what proportion of cases the mean prediction falls
    outside of the closest quantile border below and above the mean.

    Parameters
    ---------
    y: pd.Series
        series of observed true values
    pred: pd.DataFrame
        output of SamQuantileMLP.predict() function
    predict_ahead: int (default=0)
        predict ahead to evaluate

    Returns
    -------
    crossings: dict
        key is quantile, value is the observed proportion of data points below that quantile
    """

    # derive quantiles from column names
    qs = [float(c.split('_')[-1]) for c in pred.columns if 'mean' not in c]

    # make sure quantiles are sorted if they arent already:
    qs = np.sort(qs)[::-1]

    # now compute the quantile crossings
    crossings = {}
    for c in range(len(qs)-1):
        crossings['%.3f < %.3f' % (qs[c], qs[c+1])] = (
            pred['predict_lead_%d_q_' % predict_ahead + str(qs[c])] <
            pred['predict_lead_%d_q_' % predict_ahead + str(qs[c+1])]).mean()

    # and also compute where the mean falls outside the most narrow quantiles
    qs_below_mean = qs[qs < 0.5]
    q_closest_to_mean_below = qs_below_mean[0]
    crossings['mean < %.3f' % q_closest_to_mean_below] = (
        pred['predict_lead_%d_mean' % predict_ahead] <
        pred['predict_lead_%d_q_' % predict_ahead + str(q_closest_to_mean_below)]).mean()

    qs_below_mean = qs[qs > 0.5]
    q_closest_to_mean_above = qs_below_mean[-1]
    crossings['mean > %.3f' % q_closest_to_mean_above] = (
        pred['predict_lead_%d_mean' % predict_ahead] >
        pred['predict_lead_%d_q_' % predict_ahead + str(q_closest_to_mean_above)]).mean()

    return crossings
