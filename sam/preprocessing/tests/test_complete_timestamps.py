import unittest
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_equal
# Below are needed for setting up tests
from sam.preprocessing import complete_timestamps
import pandas as pd
import numpy as np


class TestCompleteTimestamps(unittest.TestCase):

    @pytest.mark.xfail(reason="Todo, see T336")
    def testCompleteTimestamps(self):
        data = pd.DataFrame({
            "TIME": pd.to_datetime(['2018/01/01 15:45:09', '2018/01/01 16:03:09',
                                    '2018/01/01 16:10:09', '2018/01/01 16:22:09']),
            "ID": 1,
            "VALUE": [1, 2, 3, 4]
        })
        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        result = complete_timestamps(data, '15min', start_time, end_time)

        # this assumes it takes the most recent previous value.
        # Since there is no previous value to 15:45:00, it is nan
        output = pd.DataFrame({
            "TIME": pd.to_datetime(['2018/01/01 15:45:00', '2018/01/01 16:00:00',
                                    '2018/01/01 16:15:00', '2018/01/01 16:30:00']),
            "ID": 1,
            "VALUE": [np.nan, 1, 3, 4]
        })
        assert_frame_equal(result, output)

        # after bfill, the nan is filled in
        output.VALUE = [1, 1, 3, 4]
        result = complete_timestamps(data, '15min', start_time, end_time, fillna_method='bfill')
        assert_frame_equal(result, output)

        # result = complete_timestamps(data, '15min', start_time, end_time, aggregate_method='sum')
        # This is a bit stupid, but i just want to make this unit test fail for now
        # This is supposed to be a test for aggregate_method
        # as it is, I have no earthly clue what that argument does.
        # So it should be slightly better documented
        # And preferably, a unit test for it should be written
        assertTrue(False)

        # result = complete_timestamps(data, '15min', start_time='', end_time='')
        # Again, I have no idea what this is supposed to have as a result...
        # unit test for this needs to be written
        assertTrue(False)

    def testIncorrectInput(self):
        data = pd.DataFrame({
            "TIME": pd.to_datetime(['2018/01/01 15:45:09', '2018/01/01 16:03:09',
                                    '2018/01/01 16:10:09', '2018/01/01 16:22:09']),
            "ID": 1,
            "VALUE": [1, 2, 3, 4]
        })
        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        # half uur is invalid timeunit
        self.assertRaises(ValueError, complete_timestamps, data, 'half uur', start_time, end_time)
        # wrong is not a time
        # integers are actually allowed, they are interpreted as UNIX time
        self.assertRaises(ValueError, complete_timestamps, data, '15min', 'wrong', end_time)

        data.columns = ["TIME", "ID", "SOMETHINGELSE"]
        # column names incorrect
        self.assertRaises(ValueError, complete_timestamps, data, '15min', start_time, end_time)

        # unknown because I have no idea what happens if you call an unknown function
        self.assertRaises(Exception, complete_timestamps, data, '15min',
                          start_time, end_time, 'unknown_fun', '')
        self.assertRaises(Exception, complete_timestamps, data, '15min',
                          start_time, end_time, '', 'unknown_fun')

if __name__ == '__main__':
    unittest.main()
