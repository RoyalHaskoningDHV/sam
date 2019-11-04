import matplotlib.pyplot as plt
import numpy as np


def diagnostic_flatline_removal(RF, raw_data, col):
    """
    Creates a diagnostic plot for the extreme value removal procedure
    in sam.preprocessing.correct_extremes.remove_extreme_values().

    Parameters:
    ----------
    RF: object
        fitted RemoveFlatlines object
    raw_data: pd.DataFrame
        non-transformed data data
    col: string
        column name to plot
    diff: pd.Series
        ab(x - rolling)
    thresh: float
        threshold used for extreme value detection

    Returns:
    -------
    fig: figure
        diagnostic plot
    """

    # get data
    x = raw_data[col].copy()
    invalid_w = np.where(RF.invalids[col])[0]
    invalid_values = x.iloc[invalid_w]

    # generate plot
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(111)
    plt.title(col)

    plt.plot(
        x.index,
        x.values,
        label='original_signal',
        lw=5)

    plt.plot(
        invalid_values.index,
        invalid_values.values,
        'o',
        ms=10,
        mew=2,
        mec='r',
        fillstyle='none',
        label='invalid samples')

    plt.legend(loc='best')
    plt.tight_layout()

    return fig
