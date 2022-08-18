import pandas as pd
from sam.validation import FlatlineValidator


def diagnostic_flatline_removal(
    flatline_validator: FlatlineValidator,
    raw_data: pd.DataFrame,
    col: str,
):
    """
    Creates a diagnostic plot for the extreme value removal procedure.

    Parameters:
    ----------
    flatline_validator: sam.validation.FlatlineValidator
        fitted FlatlineValidator object
    raw_data: pd.DataFrame
        non-transformed data
    col: string
        column name to plot

    Returns:
    -------
    fig: matplotlib.pyplot.figure
        diagnostic plot
    """
    import matplotlib.pyplot as plt

    # get data
    x = raw_data[col].copy()
    invalid_w = flatline_validator.validate(raw_data)[col]
    invalid_values = x[invalid_w]

    # generate plot
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(111)
    plt.title(col)

    plt.plot(x.index, x.values, label="original_signal", lw=5)

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
    plt.tight_layout()

    return fig
