
def compute_quantile_ratios(y, pred, predict_ahead=0):
    """
    Computes the proportion of data points beneath each quantile.

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

    quantile_ratios = {}
    for q in qs:
        # mean here computes ratio (mean of True/Falses - 0/1s)
        quantile_ratios[q] = (y < pred['predict_lead_%d_q_' % predict_ahead + str(q)]).mean()

    return quantile_ratios
