import pandas as pd
from sam.validation import MADValidator


def diagnostic_extreme_removal(
    mad_validator: MADValidator,
    raw_data: pd.DataFrame,
    col: str,
):
    """
    Creates a diagnostic plot for the extreme value removal procedure.

    Parameters:
    ----------
    mad_validator: sam.validation.MADValidator
        fitted MADValidator object
    raw_data: pd.DataFrame
        non-transformed data data
    col: string
        column name to plot

    Returns:
    -------
    fig: matplotlib.pyplot.figure
        diagnostic plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # get data
    x = raw_data[col].copy()
    invalid_w = mad_validator.validate(raw_data)[col]
    invalid_values = x.loc[invalid_w]
    rolling = mad_validator._compute_rolling(x)
    diff = x.values - rolling

    # generate plot
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.title(col)

    plt.plot(x.index, x.values, label="original_signal", lw=5)

    plt.plot(
        rolling.index,
        rolling.values,
        "--k",
        label="rolling median",
        lw=3,
    )

    plt.plot(
        invalid_values.index,
        invalid_values.values,
        "o",
        ms=10,
        mew=2,
        mec="r",
        fillstyle="none",
        label="invalid samples",
    )

    plt.legend(loc="best")
    sns.despine()

    plt.subplot(212)
    plt.plot(diff.values, label="abs(original - rolling)")
    plt.axhline(mad_validator.thresh_high[col], ls="--", c="r")
    plt.axhline(mad_validator.thresh_low[col], ls="--", c="r", label="thresholds")
    plt.legend(loc="best")
    sns.despine()
    plt.tight_layout()

    return fig
