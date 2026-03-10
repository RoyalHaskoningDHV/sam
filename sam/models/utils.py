import logging
import pandas as pd

logger = logging.getLogger(__name__)

def apply_stitching(X, y, weights, stitch_on_x=False):
    """
    Remove rows with nan that can't be used for fitting ML models based on the target

    Parameters
    ----------
    X: pd.DataFrame
        The independent variables used to 'train' the model
    y: pd.Series or pd.DataFrame
        Target data (dependent variable) used to 'train' the model.
    weights: pd.Series
        Weights for the samples, used to 'train' the model.
    stitch_on_x: bool
        If True, remove rows with nan in X and y. Otherwise, remove rows with nan in y.

    """
    X, y = X.copy(), y.copy()
    nan_rows = pd.DataFrame(y).isna().any(axis=1)
    if stitch_on_x:
        nan_rows = nan_rows | pd.DataFrame(X).isna().any(axis=1)

    logger.warning(
        "Applying stitching:\n"
        f"  • Rows with NaNs: {nan_rows.sum()}/{len(X)} ({(nan_rows.sum()/len(X))*100:.2f}%)\n"
        f"  • stitch_on_x: {stitch_on_x}"
    )

    X = X.loc[~nan_rows]
    y = y.loc[~nan_rows]
    weights = weights.loc[~nan_rows]

    return X, y, weights


def remove_until_first_value(X, y, weights):
    """
    Remove rows until the first value is available.

    Parameters
    ----------
    X: pd.DataFrame
        The independent variables used to 'train' the model
    y: pd.Series or pd.DataFrame
        Target data (dependent variable) used to 'train' the model.
    weights: pd.Series
        Weights for the samples, used to 'train' the model.

    """
    X, y, weights = X.copy(), y.copy(), weights.copy()
    first_complete_index = X.dropna(axis=0, how="any").index[0]
    X = X.loc[first_complete_index:]
    y = y.loc[first_complete_index:]
    weights = weights.loc[first_complete_index:]
    return X, y, weights
