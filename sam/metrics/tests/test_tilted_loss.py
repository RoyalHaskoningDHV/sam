import unittest

from numpy.testing import assert_almost_equal
from sam.metrics import tilted_loss


class TestTiltedLoss(unittest.TestCase):
    def test_tilted_loss(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.1, 3.1, 3.9, 4.9]

        assert_almost_equal(tilted_loss(y_true, y_pred, 0.1), 0.058)
        assert_almost_equal(tilted_loss(y_true, y_pred, 0.5), 0.05)
        assert_almost_equal(tilted_loss(y_true, y_pred, 0.9), 0.042)


if __name__ == "__main__":
    unittest.main()
