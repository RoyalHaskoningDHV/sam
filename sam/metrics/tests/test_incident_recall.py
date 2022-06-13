import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

# Below are needed for setting up tests
from sam.metrics import (
    incident_recall,
    make_incident_recall_scorer,
    precision_incident_recall_curve,
)
from sam.metrics.incident_recall import _merge_thresholds
from sklearn.base import BaseEstimator


class TestIncidentRecall(unittest.TestCase):
    def testScoring(self):
        range_pred = (2, 3)
        incidents = [0, 0, 0, 1, 1, 0, 1, 0]

        y_pred = [0, 1, 0, 0, 0, 0, 0, 0]
        # 2 out of 3 incidents were predicted, recall 2/3
        self.assertAlmostEqual(incident_recall(incidents, y_pred, range_pred), 2 / 3)

        y_pred = [1, 0, 0, 0, 0, 0, 1, 1]
        # 1 out of 3 incidents were predicted, recall 1/3
        self.assertAlmostEqual(incident_recall(incidents, y_pred, range_pred), 1 / 3)

        y_pred = [1, 1, 1, 1, 1, 1, 1, 1]
        # 3 out of 3 incidents were predicted, recall 1
        self.assertAlmostEqual(incident_recall(incidents, y_pred, range_pred), 1)

        # also with different range_pred

        range_pred = (0, 0)

        y_pred = [0, 0, 0, 0, 1, 0, 1, 0]
        # 2 out of 3, recall of 2/3
        self.assertAlmostEqual(incident_recall(incidents, y_pred, range_pred), 2 / 3)

        y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
        # 0 out of 3, recall of 0
        self.assertAlmostEqual(incident_recall(incidents, y_pred, range_pred), 0)

    def testIncorrectInputs(self):

        self.assertRaises(Exception, incident_recall, "test", "test2", "test3")
        # negative range not possible?
        self.assertRaises(Exception, incident_recall, 0, 0, 0, (-2, -1))


class TestMakeIncidentRecallScorer(unittest.TestCase):
    def testMakeScorer(self):

        op = type(
            "MyClassifier",
            (BaseEstimator, object),
            {"predict": lambda self, X: np.array([0, 1, 0, 0, 0, 0, 0, 0])},
        )
        data = pd.DataFrame({"incident": [0, 0, 0, 1, 1, 0, 1, 0], "other": 1})

        scorer = make_incident_recall_scorer((2, 3), "incident")
        self.assertTrue(callable(scorer))

        # same as the first test above, should be 2/3 just like there
        self.assertAlmostEqual(scorer(op(), data), 2 / 3)

        wrongdata = pd.DataFrame({"other": [1, 1, 1], "misc": [2, 2, 2]})
        self.assertRaises(KeyError, scorer, op(), wrongdata)


class TestIncidentPrecisionRecallCurve(unittest.TestCase):
    def testCurve(self):
        y_incidents = [0, 0, 0, 1]
        y_pred = [0.1, 0.2, 0.3, 0.4]
        p, r, t = precision_incident_recall_curve(y_incidents, y_pred, range_pred=(0, 1))
        assert_array_almost_equal(p, np.array([1, 1, 1]))
        assert_array_equal(r, np.array([1, 1, 0]))
        assert_array_equal(t, np.array([0.3, 0.4]))

    def testCurveAgain(self):
        y_pred = np.array([0.4, 0.4, 0.1, 0.2, 0.6, 0.5, 0.1])
        y_incidents = np.array([0, 1, 0, 0, 0, 0, 1])
        range_pred = (0, 2)
        p, r, t = precision_incident_recall_curve(y_incidents, y_pred, range_pred)
        assert_array_equal(p, np.array([5 / 7, 0.8, 1, 1, 1, 1]))
        assert_array_equal(r, np.array([1, 1, 1, 0.5, 0.5, 0]))
        assert_array_equal(t, np.array([0.1, 0.2, 0.4, 0.5, 0.6]))

    def testIncorrectInput(self):
        self.assertRaises(Exception, precision_incident_recall_curve, "test", "test2", (0, 0))
        # negative range not possible
        self.assertRaises(
            Exception, precision_incident_recall_curve, [0, 0, 1], [1, 1, 1], (-2, -1)
        )


class TestMergeThresholds(unittest.TestCase):
    def testMergeThresholdNormal(self):
        p, r, t = _merge_thresholds(
            [0.1, 0.2, 0.5, 0.6, 0.8, 1],
            [0, 0.3, 0.5, 0.8, 0.9, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14, 15, 16],
        )
        expected_t = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1])
        expected_p = np.array([1, 1, 2, 3, 3, 4, 5, 6, 6, 7])
        expected_r = np.array([10, 11, 11, 11, 12, 13, 13, 14, 15, 16])
        assert_array_equal(p, expected_p)
        assert_array_equal(r, expected_r)
        assert_array_equal(t, expected_t)

    def testMergeThresholdEmpty(self):
        p, r, t = _merge_thresholds([], [0.5], [0.7], [0.8, 0.9])
        assert_array_equal(t, np.array([0.5]))
        assert_array_equal(p, np.array([0.7, 0.7]))
        assert_array_equal(r, np.array([0.8, 0.9]))

    def testMergeThresholdEqual(self):
        p, r, t = _merge_thresholds([0.5], [0.5], [1, 1], [0.6, 0.6])
        assert_array_equal(t, np.array([0.5]))
        assert_array_equal(p, np.array([1, 1]))
        assert_array_equal(r, np.array([0.6, 0.6]))


if __name__ == "__main__":
    unittest.main()
