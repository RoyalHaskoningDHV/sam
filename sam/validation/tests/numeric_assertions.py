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
        except:
            self.fail(self._formatMessage(msg, standardMsg))

    def assertAllNotNaN(self, value, msg=None):
        """
        Fail if provided value is NaN
        """
        standardMsg = "There is at least 1 NaN in provided series"
        try:
            if np.all(~np.isnan(value)):
                self.fail(self._formatMessage(msg, standardMsg))
        except:
            pass
