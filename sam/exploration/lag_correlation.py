import logging
from typing import Callable, Union

import numpy as np
import pandas as pd
from sam.feature_engineering.rolling_features import BuildRollingFeatures
from sam.logging_functions import log_dataframe_characteristics

logger = logging.getLogger(__name__)


def lag_correlation(
    df: pd.DataFrame,
    target_name: str,
    lag: int = 12,
    method: Union[str, Callable] = "pearson",
):
    """
    Creates a new dataframe that contains the correlation of target_name
    with other variables in the dataframe based on the output from
    BuildRollingFeatures. The results are processed for easy visualization,
    with a column for the lag and then correlation per feature.

    Parameters
    ----------
    df: pd.DataFrame
        input dataframe contains variables to calculate lag correlation of
    target_name: str
        The name of the goal variable to calculate lag correlation with
    lag: int or list of ints (default=12)
        When an integer is provided, a range is created from 0 to lag in steps
        of 1, when an array of ints is provided, this is directly used.
        Default is 12, which means the correlation is calculated for lag
        ranging from 0 to 11.
    method: string or callable, optional (default='pearson')
        The method used to calculate correlation. See pandas.DataFrame.corrwith.
        Options are {‘pearson’, ‘kendall’, ‘spearman’}, or a callable.

    Returns
    -------
    tab: pandas dataframe
        A dataframe with the correlations and shift.
        The column header contains the feature name.

    Examples
    --------
    >>> import pandas as pd
    >>> from sam.exploration import lag_correlation
    >>> import numpy as np
    >>> X = pd.DataFrame({
    ...        'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0,
    ...                 0.0, 0.0, 0.0, 0.0, 0.0],
    ...        'DEBIET#A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
    ...        'DEBIET#B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
    >>> X['DEBIET#TOTAAL'] = X['DEBIET#A'] + X['DEBIET#B']
    >>> tab = lag_correlation(X, 'DEBIET#TOTAAL', lag=11)
    >>> tab
        LAG  DEBIET#A  DEBIET#B      RAIN
    0     0  0.838591  0.897340 -0.017557
    1     1  0.436484  0.102808  0.204983
    2     2  0.287863 -0.401768  0.672316
    3     3 -0.388095 -0.140876  0.188438
    4     4 -0.632980 -0.509307 -0.227071
    5     5 -0.667537 -0.367268 -0.048162
    6     6 -0.152832  0.615239  0.110876
    7     7  0.457496 -0.107833 -0.719702
    8     8  0.291111  0.039253  0.871695
    9     9  0.188982  0.755929 -0.944911
    10   10  1.000000 -1.000000  1.000000
    """

    logging.debug("Now creating lag correlation with lag {}".format(lag))
    y = df[target_name]
    X = df.drop(target_name, axis=1)

    if np.isscalar(lag):
        lag = np.arange(lag)

    RollingFeatures = BuildRollingFeatures(
        rolling_type="lag", window_size=lag, lookback=0, keep_original=False
    )
    df = RollingFeatures.fit_transform(X)
    corr_table = df.corrwith(y, method=method).reset_index()
    corr_table.columns = ["index", "corr"]
    corr_table["LAG"] = corr_table["index"].apply(lambda x: x.rsplit("_", 1)[1])
    corr_table["GROUP"] = corr_table["index"].apply(lambda x: x.rsplit("#", 1)[0])

    tab = pd.pivot_table(corr_table, values="corr", index="LAG", columns="GROUP").reset_index()
    tab["LAG"] = pd.to_numeric(tab["LAG"])
    tab.columns.name = None
    tab = tab.sort_values("LAG").reset_index(drop=True)

    logging.info("create_lag_correlation output:")
    log_dataframe_characteristics(tab, logging.INFO)
    return tab
