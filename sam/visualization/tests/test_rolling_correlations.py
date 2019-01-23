import unittest
import pytest
from sam.visualization import plot_lag_correlation
import pandas as pd
from sam.feature_engineering import BuildRollingFeatures
from sam.feature_selection import create_lag_correlation
from sam.visualization import plot_lag_correlation
import numpy as np


class TestRollingCorrelations(unittest.TestCase):

    @pytest.mark.mpl_image_compare(tolerance=20)
    def test_rolling_correlations(self):

        goal_feature = 'TOTAAL_lag_0'
        df = pd.DataFrame({
                       'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       'A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
                       'B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
        df['TOTAAL'] = df['A'] + df['B']
        RollingFeatures = BuildRollingFeatures(rolling_type='lag',
                                               window_size=np.arange(12),
                                               lookback=0, keep_original=False)
        res = RollingFeatures.fit_transform(df)
        test = create_lag_correlation(res, goal_feature)
        ax = plot_lag_correlation(test)
        return ax.get_figure()

if __name__ == '__main__':
    unittest.main()
