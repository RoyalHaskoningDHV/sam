import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sam.preprocessing import average_winter_time, label_dst


class TestLabelDst(unittest.TestCase):
    def test_summertime(self):
        time1 = "2019/03/31 01:45:00"
        time2 = "2019/03/31 03:00:00"
        freq = "15min"
        daterange = pd.date_range(time1, time2, freq=freq)

        result = label_dst(pd.Series(daterange))
        # 1:45 is normal, 2:00,15,30,45 are to_summertime, 3:00 is normal
        assert_array_equal(
            result,
            np.array(
                [
                    "normal",
                    "to_summertime",
                    "to_summertime",
                    "to_summertime",
                    "to_summertime",
                    "normal",
                ]
            ),
        )

    def test_wintertime(self):
        time1 = "2019/10/27 01:45:00"
        time2 = "2019/10/27 03:00:00"
        freq = "15min"
        daterange = pd.date_range(time1, time2, freq=freq)

        result = label_dst(pd.Series(daterange))
        # 1:45 is normal, 2:00,15,30,45 are to_wintertime, 3:00 is normal
        assert_array_equal(
            result,
            np.array(
                [
                    "normal",
                    "to_wintertime",
                    "to_wintertime",
                    "to_wintertime",
                    "to_wintertime",
                    "normal",
                ]
            ),
        )

    def test_incorrect(self):
        # this can either be a valueerror or an attributeerror
        self.assertRaises(Exception, label_dst, 10)
        self.assertRaises(Exception, label_dst, [7])
        self.assertRaises(Exception, label_dst, {"TIME": "2019/10/27 01:45:00"})
        self.assertRaises(Exception, label_dst, np.datetime64("2005-02-25"))


class TestAverageWintertime(unittest.TestCase):
    def test_wintertime(self):
        time1 = "2019/10/27 01:45:00"
        time2 = "2019/10/27 03:00:00"
        freq = "15min"
        daterange = pd.date_range(time1, time2, freq=freq)

        test_df = pd.DataFrame(
            {
                "TIME": daterange.values[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]],
                "VALUE": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "ID": np.ones(10),
            },
            columns=["TIME", "ID", "VALUE"],
        )
        test_df_copy = test_df.copy()

        output_df = pd.DataFrame(
            {
                "TIME": daterange.values[[0, 1, 2, 3, 4, 5]],
                "VALUE": np.array([0, 1.5, 3.5, 5.5, 7.5, 9]),
                "ID": np.ones(6),
            },
            columns=["TIME", "ID", "VALUE"],
        )

        assert_frame_equal(average_winter_time(test_df), output_df)
        # The original frame cannot have changed because of the call
        assert_frame_equal(test_df, test_df_copy)

        # Cannot use already existing tmp col
        self.assertRaises(Exception, average_winter_time, test_df, "ID")

    def test_incorrect_inputs(self):
        incomplete_df = pd.DataFrame({"VALUE": np.array([1, 2, 3])})
        self.assertRaises(Exception, average_winter_time, 7)
        self.assertRaises(Exception, average_winter_time, [1, 2, 3])
        self.assertRaises(Exception, average_winter_time, incomplete_df)


if __name__ == "__main__":
    unittest.main()
