import unittest

import numpy as np
import pandas as pd
import pytest
from sam.visualization import plot_feature_importances


class TestFeatureImportancePlot(unittest.TestCase):
    def setUp(self):

        base_imps = np.array([0.18, 0.2, 0.19, 0.21, 0.22])
        imps = pd.DataFrame(
            {
                "feature_1#lag_1": base_imps * 2,
                "feature_1#mean_4": base_imps,
                "feature_2#lag_1": base_imps,
                "feature_2#mean_4": base_imps / 2,
            }
        )

        self.f, self.f_sum = plot_feature_importances(imps, ["feature_1", "feature_2"])

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_importance_barplot(self):
        return self.f

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_importance_barplot_sum(self):
        return self.f_sum


if __name__ == "__main__":
    unittest.main()
