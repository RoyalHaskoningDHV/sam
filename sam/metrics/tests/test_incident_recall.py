import unittest
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_equal
# Below are needed for setting up tests
from sam.metrics import incident_recall, make_incident_recall_scorer
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np


class TestIncidentRecall(unittest.TestCase):

    def testScoring(self):
        range_pred = (2, 3)
        incidents = [0, 0, 0, 1, 1, 0, 1, 0]
        y_true = [1, 1, 1, 1, 1, 0, 0, 0]

        y_pred = [0, 1, 0, 0, 0, 0, 0, 0]
        # 2 out of 3 incidents were predicted, recall 2/3
        self.assertAlmostEqual(incident_recall(y_true, y_pred, incidents, range_pred), 2/3)

        y_pred = [1, 0, 0, 0, 0, 0, 1, 1]
        # 1 out of 3 incidents were predicted, recall 1/3
        self.assertAlmostEqual(incident_recall(y_true, y_pred, incidents, range_pred), 1/3)

        y_pred = [1, 1, 1, 1, 1, 1, 1, 1]
        # 3 out of 3 incidents were predicted, recall 1
        self.assertAlmostEqual(incident_recall(y_true, y_pred, incidents, range_pred), 1)

        # also with different range_pred

        range_pred = (0, 0)
        y_true = incidents

        y_pred = [0, 0, 0, 0, 1, 0, 1, 0]
        # 2 out of 3, recall of 2/3
        self.assertAlmostEqual(incident_recall(y_true, y_pred, incidents, range_pred), 2/3)

        y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
        # 0 out of 3, recall of 0
        self.assertAlmostEqual(incident_recall(y_true, y_pred, incidents, range_pred), 0)

    def testIncorrectInputs(self):

        self.assertRaises(Exception, incident_recall, "test", "test2", "test3")
        # negative range not possible?
        self.assertRaises(Exception, incident_recall, 0, 0, 0, (-2, -1))


class TestMakeIncidentRecallScorer(unittest.TestCase):

    def testMakeScorer(self):

        op = type("MyClassifier", (BaseEstimator, object),
                  {"predict": lambda self, X: np.array([0, 1, 0, 0, 0, 0, 0, 0])})
        data = pd.DataFrame({"incident": [0, 0, 0, 1, 1, 0, 1, 0], "other": 1})
        y_true = [1, 1, 1, 1, 1, 0, 0, 0]

        scorer = make_incident_recall_scorer((2, 3), "incident")
        self.assertTrue(callable(scorer))

        # same as the first test above, should be 2/3 just like there
        self.assertAlmostEqual(scorer(op(), data, y_true), 2/3)

        wrongdata = pd.DataFrame({"other": [1, 1, 1], "misc": [2, 2, 2]})
        self.assertRaises(KeyError, scorer, op(), wrongdata, y_true)


if __name__ == '__main__':
    unittest.main()
