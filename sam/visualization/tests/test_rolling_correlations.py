import unittest

import pandas as pd
import pytest
from sam.exploration import lag_correlation
from sam.visualization import plot_lag_correlation


class TestRollingCorrelations(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_rolling_correlations(self):

        X = pd.DataFrame(
            {
                "RAIN": [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "A": [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
                "B": [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5],
            }
        )
        X["TOTAAL"] = X["A"] + X["B"]
        test = lag_correlation(X, "TOTAAL")
        ax = plot_lag_correlation(test)
        return ax.get_figure()


if __name__ == "__main__":
    unittest.main()
