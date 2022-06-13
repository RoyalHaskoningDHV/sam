import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

# Below are needed for setting up tests
from sam.exploration import incident_curves, incident_curves_information


class TestFindIncidentCurves(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "ACTUAL": [0.3, np.nan, 0.3, np.nan, 0.3, 0.5, np.nan, 0.7],
                "PREDICT_HIGH": 0.6,
                "PREDICT_LOW": 0.4,
            }
        )
        self.data_bak = self.data.copy()

    def test_no_gaps(self):
        result = incident_curves(self.data)
        assert_array_equal(result, np.array([1, 0, 2, 0, 3, 0, 0, 4]))
        assert_frame_equal(self.data, self.data_bak)

    def test_gap(self):
        result = incident_curves(self.data, max_gap=1)
        assert_array_equal(result, np.array([1, 1, 1, 1, 1, 0, 0, 2]))
        assert_frame_equal(self.data, self.data_bak)

    def test_condition(self):
        result = incident_curves(self.data, max_gap=1, max_gap_perc=0.2)
        assert_array_equal(result, np.array([0, 0, 0, 0, 0, 0, 0, 2]))
        assert_frame_equal(self.data, self.data_bak)

    def test_nooutliers(self):
        data = pd.DataFrame(
            {
                "ACTUAL": [0.5, 0.5, 0.5, 0.5, 0.5],
                "PREDICT_HIGH": 0.6,
                "PREDICT_LOW": 0.4,
            }
        )
        result = incident_curves(data, max_gap=1)
        assert_array_equal(result, np.array([0, 0, 0, 0, 0]))

    def test_gap_edge(self):
        data = pd.DataFrame({"ACTUAL": [0.5, 1, 0.5], "PREDICT_HIGH": 0.6, "PREDICT_LOW": 0.4})
        result = incident_curves(data, max_gap=1)
        assert_array_equal(result, np.array([0, 1, 0]))

    def test_2gap_edge(self):
        data = pd.DataFrame(
            {"ACTUAL": [0.5, 0.5, 1, 0.5, 0.5], "PREDICT_HIGH": 0.6, "PREDICT_LOW": 0.4}
        )
        result = incident_curves(data, max_gap=2)
        assert_array_equal(result, np.array([0, 0, 1, 0, 0]))

    def test_wrong_input(self):
        wrongdata = pd.DataFrame({"AGTUAL": [0.5, 0.3], "PREDICT_HIGH": 0.6, "PREDICT_LOW": 0.4})
        self.assertRaises(Exception, incident_curves, wrongdata)

        with self.assertRaises(Exception):
            incident_curves(self.data, max_gap_perc=None)

    def test_higher_index(self):
        # Test issue T354
        new_index = self.data.copy()
        new_index.index = range(9500, 9508)
        result = incident_curves(new_index)
        assert_array_equal(result, np.array([1, 0, 2, 0, 3, 0, 0, 4]))


class TestCreateIncidentInformation(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "TIME": range(1547477436, 1547477436 + 3),  # unix timestamps
                "ACTUAL": [0.3, 0.5, 0.7],
                "PREDICT_HIGH": 0.6,
                "PREDICT_LOW": 0.4,
                "PREDICT": 0.5,
            },
            columns=["ACTUAL", "PREDICT", "PREDICT_HIGH", "PREDICT_LOW", "TIME"],
        )
        self.no_outliers = pd.DataFrame(
            {
                "TIME": range(1547477436, 1547477436 + 3),
                "ACTUAL": [0.5, 0.5, 0.5],
                "PREDICT_HIGH": 0.6,
                "PREDICT_LOW": 0.4,
                "PREDICT": 0.5,
            },
            columns=["ACTUAL", "PREDICT", "PREDICT_HIGH", "PREDICT_LOW", "TIME"],
        )

    def test_aggregated(self):
        result = incident_curves_information(self.data)
        expected = pd.DataFrame(
            {
                "OUTLIER_CURVE": [1, 2],
                "OUTLIER_DURATION": [1, 1],
                "OUTLIER_START_TIME": [1547477436, 1547477438],
                "OUTLIER_END_TIME": [1547477436, 1547477438],
                "OUTLIER_SCORE_MAX": 1 / 11,
                "OUTLIER_DIST_SUM": 0.1,
                "OUTLIER_DIST_MAX": 0.1,
                "OUTLIER_TYPE": ["negative", "positive"],
            },
            columns=[
                "OUTLIER_CURVE",
                "OUTLIER_DURATION",
                "OUTLIER_TYPE",
                "OUTLIER_SCORE_MAX",
                "OUTLIER_START_TIME",
                "OUTLIER_END_TIME",
                "OUTLIER_DIST_SUM",
                "OUTLIER_DIST_MAX",
            ],
        ).set_index("OUTLIER_CURVE")
        assert_frame_equal(result, expected)

        # Alternative test for if there are no outliers. We re-use expected, but throw away
        # All the rows, because there shouldn't be any outlier_curves
        expected_no_outliers = expected.iloc[[]]
        result_no_outliers = incident_curves_information(self.no_outliers)
        assert_frame_equal(result_no_outliers, expected_no_outliers)

    def test_not_aggregated(self):
        result = incident_curves_information(self.data, return_aggregated=False)
        expected = pd.DataFrame(
            {
                "ACTUAL": [0.3, 0.5, 0.7],
                "PREDICT": 0.5,
                "PREDICT_HIGH": 0.6,
                "PREDICT_LOW": 0.4,
                "TIME": range(1547477436, 1547477436 + 3),
                "OUTLIER_CURVE": [1, 0, 2],
                "OUTLIER": [True, False, True],
                "OUTLIER_DIST": [0.1, 0, 0.1],
                "OUTLIER_SCORE": [1 / 11, 0, 1 / 11],
                "OUTLIER_TYPE": ["negative", "none", "positive"],
            },
            columns=[
                "ACTUAL",
                "PREDICT",
                "PREDICT_HIGH",
                "PREDICT_LOW",
                "TIME",
                "OUTLIER_CURVE",
                "OUTLIER",
                "OUTLIER_DIST",
                "OUTLIER_SCORE",
                "OUTLIER_TYPE",
            ],
        )
        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
