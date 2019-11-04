import matplotlib.pyplot as plt
import numpy as np


def diagnostic_extreme_removal(REV, raw_data, col):
    """
    Creates a diagnostic plot for the extreme value removal procedure
    in sam.preprocessing.correct_extremes.remove_extreme_values().

    Parameters:
    ----------
    REV: object
        fitted RemoveExtremeValues object
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
    f: figure
        diagnostic plot
    """

    import seaborn as sns

    # get data
    x = raw_data[col].copy()
    invalid_w = np.where(REV.invalids[col])[0]
    invalid_values = x.iloc[invalid_w]

    # generate plot
    f = plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.title(col)

    plt.plot(
        x.index,
        x.values,
        label='original_signal',
        lw=5)

    plt.plot(
        REV.rollings[col].index,
        REV.rollings[col].values,
        '--k',
        label='rolling median',
        lw=3)

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
    sns.despine()

    plt.subplot(212)
    plt.plot(REV.diffs[col].values, label='abs(original - rolling)')
    plt.axhline(REV.thresh_high[col], ls='--', c='r')
    plt.axhline(REV.thresh_low[col], ls='--', c='r', label='thresholds')
    plt.legend(loc='best')
    sns.despine()
    plt.tight_layout()

    return f
