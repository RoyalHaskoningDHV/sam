import pytest
import numpy as np
import pandas as pd

from ..signalaligner import SignalAligner


@pytest.mark.parametrize("N_aligned", [40, 50])
@pytest.mark.parametrize("N1", [60, 90, 120])
@pytest.mark.parametrize("N2", [60, 90, 120])
@pytest.mark.parametrize("reference", [None, 0, 1])
def test_signalaligner(N_aligned, N1, N2, reference):

    lat = np.random.randn(N_aligned) + np.random.standard_normal(N_aligned)

    lat1 = np.random.randn(N1)
    i1 = np.random.randint(N1 - N_aligned)
    lat1[i1 : i1 + N_aligned] = lat

    lat2 = np.random.randn(N2)
    i2 = np.random.randint(N2 - N_aligned)
    lat2[i2 : i2 + N_aligned] = lat

    df1 = pd.DataFrame({"data": np.random.randn(N1), "lat": lat1})

    df2 = pd.DataFrame({"data": np.random.randn(N2), "lat": lat2})

    col1 = "lat"
    col2 = "lat"

    sa = SignalAligner()

    df, offset = sa.align_dataframes(df1, df2, col1, col2, reference=reference)

    N_equal = sum([1 if x1 == x2 else 0 for x1, x2 in zip(df["lat_x"].values, df["lat_y"].values)])

    assert (
        N_equal == N_aligned
    ), "Unexpectedly unequal length of aligned signal and overlapping signal"

    if reference is None:
        assert (df.shape[0] >= df1.shape[0]) and (df.shape[0] >= df2.shape[0])
    elif reference == 0:
        assert df.shape[0] == df1.shape[0]
        assert df.reset_index().loc[:, col1 + "_x"].equals(df1.reset_index().loc[:, col1])
    elif reference == 1:
        assert df.shape[0] == df2.shape[0]
        assert df.reset_index().loc[:, col2 + "_y"].equals(df2.reset_index().loc[:, col2])
