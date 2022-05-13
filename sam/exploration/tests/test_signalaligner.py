import pytest
import numpy as np
import pandas as pd

from ..signalaligner import SignalAligner


N_aligned_list = [40, 50, 60]
N1_list = [60, 90, 120]
N2_list = [60, 90, 120]
params = []
for N_aligned in N_aligned_list:
    for N1 in N1_list:
        for N2 in N2_list:
            params.append((N_aligned, N1, N2))


@pytest.mark.parametrize("N_aligned, N1, N2", params)
def test_signalaligner(N_aligned, N1, N2):

    lat = np.random.randn(N_aligned) + np.random.standard_normal(N_aligned)

    lat1 = np.random.randn(N1)
    i1 = np.random.randint(N1 - N_aligned)
    lat1[i1: i1 + N_aligned] = lat

    lat2 = np.random.randn(N2)
    i2 = np.random.randint(N2 - N_aligned)
    lat2[i2: i2 + N_aligned] = lat


    df1 = pd.DataFrame({
        'data': np.random.randn(N1),
        'lat': lat1,
    })

    df2 = pd.DataFrame({
        'data': np.random.randn(N2),
        'lat': lat2,
    })

    col1 = 'lat'
    col2 = 'lat'

    sa = SignalAligner()

    offset, df1, df2 = sa.align_dataframes(df1, df2, col1, col2)

    N_equal = sum(
        [1 if x1 == x2 else 0 for x1, x2 in zip(df1['lat'].values, df2['lat'].values)]
    )

    assert(N_equal == N_aligned), (
        "Unexpectedly unequal length of aligned signal and overlapping signal"
    )