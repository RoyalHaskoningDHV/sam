import unittest
import pytest
from sam.visualization import make_precision_recall_curve
import numpy as np
import matplotlib


class TestPrecisionRecall(unittest.TestCase):

    @pytest.mark.mpl_image_compare(tolerance=20)
    def test_regular_precision_recall_plot(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax = make_precision_recall_curve(y_true, y_scores)
        return ax.get_figure()

    @pytest.mark.mpl_image_compare(tolerance=20)
    def test_incident_precision_recall_plot(self):
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y_incidents = np.array([0, 0, 1, 0, 0, 1])
        ax = make_precision_recall_curve(y_incidents, y_scores, (0, 1))
        return ax.get_figure()

if __name__ == '__main__':
    unittest.main()
