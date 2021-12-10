import unittest

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sam.visualization import performance_evaluation_fixed_predict_ahead


class TestPerformanceEvaluation(unittest.TestCase):
    def setUp(self):

        # setup model as set of sinewaves
        np.random.seed(42)
        N = 1000
        model = np.zeros(N)
        for f in [0.001, 0.005, 0.01, 0.05]:  # frequencies (Hz)
            for p in [0, np.pi / 4]:  # phase offsets (radians)
                model += (1 / f) * np.sin(2 * np.pi * f * np.arange(N) + p)

        # data is model plus some noise
        data = model + np.random.normal(scale=1000, size=N)

        # split in train and test sets
        times = pd.date_range("1/1/2011", periods=N, freq="5 min")
        data = pd.Series(data, index=times)
        model = pd.DataFrame(model, columns=["predict_lead_0_mean"], index=times)

        train_prop = 0.7
        y_hat_test = model[int(N * train_prop) :]
        y_true_test = data[int(N * train_prop) :]
        y_hat_train = model[: int(N * train_prop)]
        y_true_train = data[: int(N * train_prop)]

        (
            self.r2_df,
            self.bar_fig,
            self.scatter_fig,
            best_res,
        ) = performance_evaluation_fixed_predict_ahead(
            y_true_train,
            y_hat_train,
            y_true_test,
            y_hat_test,
            resolutions=[None, "15min", "1H", "1D", "1W"],
        )

    def test_performance_evaluation_fixed_predict_ahead_r2_df(self):

        expected_df = pd.DataFrame(
            {
                "R2": [
                    65.6654375,
                    69.3605544,
                    86.1415235,
                    86.6202232,
                    96.7051635,
                    95.3895701,
                    99.9565576,
                    99.0940139,
                    99.93533776274832,
                    96.70930510331821,
                ],
                "dataset": [
                    "train",
                    "test",
                    "train",
                    "test",
                    "train",
                    "test",
                    "train",
                    "test",
                    "train",
                    "test",
                ],
                "resolution": [
                    "native",
                    "native",
                    "15min",
                    "15min",
                    "1H",
                    "1H",
                    "1D",
                    "1D",
                    "1W",
                    "1W",
                ],
            }
        )

        assert_frame_equal(expected_df, self.r2_df)

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_performance_evaluation_fixed_predict_ahead_bar_fig(self):
        return self.bar_fig

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_performance_evaluation_fixed_predict_ahead_scatter_fig(self):
        return self.scatter_fig


if __name__ == "__main__":
    unittest.main()
