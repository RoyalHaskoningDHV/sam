import pandas as pd


def remove_target_nan(X, y, weights, use_x=False):
    """
    Remove rows with nan that can't be used for fitting ML models

    Parameters
    ----------
    X: pd.DataFrame
        The independent variables used to 'train' the model
    y: pd.Series or pd.DataFrame
        Target data (dependent variable) used to 'train' the model.
    weights: pd.Series
        Weights for the samples, used to 'train' the model.
    use_x: bool
        If True, remove rows with nan in X and y. Otherwise, remove rows with nan in y.

    """
    X, y = X.copy(), y.copy()
    targetnanrows = pd.DataFrame(y).isna().any(axis=1)
    if use_x:
        targetnanrows = targetnanrows | pd.DataFrame(X).isna().any(axis=1)
    X = X.loc[~targetnanrows]
    y = y.loc[~targetnanrows]
    weights = weights.loc[~targetnanrows]

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
