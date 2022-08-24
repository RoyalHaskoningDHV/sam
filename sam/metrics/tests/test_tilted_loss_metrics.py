import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sam.metrics import joint_mae_tilted_loss, joint_mse_tilted_loss, tilted_loss


class TestTiltedLoss(unittest.TestCase):
    def setUp(self) -> None:
        actual_data = {
            "output_1": [1, 2, 3, 4, 5],
            "output_2": [1, 3, 5, 7, 9],
        }
        self.y_true_df = pd.DataFrame(data=actual_data)

        pred_data = {
            "output_1_quantile_1": [0.9, 1.9, 2.9, 3.9, 4.9],
            "output_2_quantile_1": [0.5, 2.5, 4.5, 6.5, 8.5],
            "output_1_quantile_2": [1.1, 2.1, 3.1, 4.1, 5.1],
            "output_2_quantile_2": [1.5, 3.5, 5.5, 7.5, 9.5],
            "output_1_mean": [0.9, 2.1, 2.9, 4.1, 4.9],
            "output_2_mean": [1.1, 2.9, 5.1, 6.9, 9.1],
        }
        self.y_pred_df = pd.DataFrame(data=pred_data)

        self.quantiles = [0.3, 0.7]
        self.n_targets = 2

        return super().setUp()

    def test_tilted_loss(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 3.9, 4.9]

        assert_almost_equal(tilted_loss(y_true, y_pred, 0.1), 0.058)
        assert_almost_equal(tilted_loss(y_true, y_pred, 0.5), 0.05)
        assert_almost_equal(tilted_loss(y_true, y_pred, 0.9), 0.042)

        # MAE should be equivalent to tilted loss and only vary by a constant factor 2
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        assert_almost_equal(mae / 2, tilted_loss(y_true, y_pred, 0.5))

    def test_joint_mae_tilted_loss(self):
        assert_almost_equal(
            joint_mae_tilted_loss(
                self.y_true_df, self.y_pred_df, quantiles=self.quantiles, n_targets=self.n_targets
            ),
            0.56,
        )

    def test_joint_mse_tilted_loss(self):
        assert_almost_equal(
            joint_mse_tilted_loss(
                self.y_true_df, self.y_pred_df, quantiles=self.quantiles, n_targets=self.n_targets
            ),
            0.38,
        )


if __name__ == "__main__":
    unittest.main()
