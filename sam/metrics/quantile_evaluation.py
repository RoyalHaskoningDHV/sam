
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
    qs = [float(c.split('_')[-1]) for c in pred.columns if not 'mean' in c]

    quantile_ratios = {
        # mean here computes ratio (mean of True/Falses - 0/1s)
        q: (y < pred['predict_lead_%d_q_' % predict_ahead + str(q)]).mean()
        for q in qs
    }

    return quantile_ratios
