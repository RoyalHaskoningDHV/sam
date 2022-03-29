import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from sam.models.base_model import BaseTimeseriesRegressor


class TestMakePredictionMonotonic(unittest.TestCase):
    def setUp(self):
        """
        Quantiles are not necessarily monotonic, however we may want to force this behavior.
        Create a dataframe with 6 mirrored quantiles + a median quantile and a mean column for
        a predict_ahead=0. Also add quantiles for predict_ahead=1 to test the grouping.
        """

        self.prediction = pd.DataFrame(
            {
                "predict_lead_0_q_2.866515719235352e-07": [-5, -5, -5],
                "predict_lead_0_q_3.167124183311998e-05": [-4, -4, -4],
                "predict_lead_0_q_0.0002326290790355401": [-3, -3, -3],
                "predict_lead_0_q_0.0013498980316301035": [-2, -2, -2],
                "predict_lead_0_q_0.02275013194817921": [-1, -1, -1],
                "predict_lead_0_q_0.15865525393145707": [0, 0, 0],
                "predict_lead_0_q_0.5": [-10, 10, -10],
                "predict_lead_0_q_0.8413447460685429": [1, 1, 1],
                "predict_lead_0_q_0.9772498680518208": [2, 2, 2],
                "predict_lead_0_q_0.9986501019683699": [3, 3, 3],
                "predict_lead_0_q_0.9997673709209645": [4, 4, 4],
                "predict_lead_0_q_0.9999683287581669": [5, 5, 5],
                "predict_lead_0_q_0.9999997133484281": [6, 6, 6],
                "predict_lead_0_mean": [0, 0, 0],
                "predict_lead_1_q_0.8413447460685429": [3, 3, 3],
                "predict_lead_1_q_0.9772498680518208": [5, 5, 5],
                "predict_lead_1_q_0.5": [4, 4, 4],
                "predict_lead_1_median": [0, 0, 0],
                "predict_lead_2_q_0.6": [2, 2, 2],
            },
            dtype=float,
        )

    def test_already_monotonic(self):
        """Sanity check if default params work
        Quantiles are already monotonic, so prediction should stay the same"""

        result = BaseTimeseriesRegressor.make_prediction_monotonic(self.prediction)
        assert_frame_equal(result, self.prediction)

    def test_not_monotonic_decreasing_lower_quantiles(self):
        """Check that we force lower quantiles (<0.5) to be monotonic decreasing
        The quantile breaking the monotonicity should propagate to the larger quantiles
        """
        row = 1
        prediction = self.prediction.copy()
        prediction["predict_lead_0_q_0.15865525393145707"][row] = -6

        result = BaseTimeseriesRegressor.make_prediction_monotonic(prediction)

        expected = self.prediction.copy()
        expected.loc[
            row,
            [
                "predict_lead_0_q_2.866515719235352e-07",
                "predict_lead_0_q_3.167124183311998e-05",
                "predict_lead_0_q_0.0002326290790355401",
                "predict_lead_0_q_0.0013498980316301035",
                "predict_lead_0_q_0.02275013194817921",
                "predict_lead_0_q_0.15865525393145707",
            ],
        ] = -6
        assert_frame_equal(result, expected)

    def test_not_monotonic_increasing_upper_quantiles(self):
        """Check that we force upper quantiles (>0.5) to be monotonic increasing
        The quantile breaking the monotonicity should propagate to the larger quantiles, also the
        quantiles from another predict_ahead value should not have influence.
        """
        row = 2
        prediction = self.prediction.copy()
        prediction["predict_lead_0_q_0.8413447460685429"][row] = 6
        prediction["predict_lead_1_q_0.8413447460685429"][row] = 100

        result = BaseTimeseriesRegressor.make_prediction_monotonic(prediction)

        expected = self.prediction.copy()
        expected.loc[
            row,
            [
                "predict_lead_0_q_0.8413447460685429",
                "predict_lead_0_q_0.9772498680518208",
                "predict_lead_0_q_0.9986501019683699",
                "predict_lead_0_q_0.9997673709209645",
                "predict_lead_0_q_0.9999683287581669",
                "predict_lead_0_q_0.9999997133484281",
            ],
        ] = 6
        expected.loc[
            row,
            [
                "predict_lead_1_q_0.8413447460685429",
                "predict_lead_1_q_0.9772498680518208",
            ],
        ] = 100
        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
