import numpy as np


class NumericAssertions:
    """
    This class is following the UnitTest naming conventions.
    It is meant to be used along with unittest.TestCase like so:

    class MyTest(unittest.TestCase, NumericAssertions):
        ...
    It needs python >= 2.6
    """

    def assertAllNaN(self, value, msg=None):
        """
        Fail if provided value is not NaN
        """
        standardMsg = "Not all values are NaN"
        try:
            if not np.all(np.isnan(value)):
                self.fail(self._formatMessage(msg, standardMsg))
        except Exception:
            self.fail(self._formatMessage(msg, standardMsg))

    def assertAllNotNaN(self, value, msg=None):
        """
        Fail if provided value contains any NaN
        """
        standardMsg = "At least one value is NaN"
        try:
            if np.any(np.isnan(value)):
                self.fail(self._formatMessage(msg, standardMsg))
        except Exception:
            self.fail(self._formatMessage(msg, standardMsg))
