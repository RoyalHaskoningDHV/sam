import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from sam.preprocessing import sam_format_to_wide, wide_to_sam_format


class TestSamReshape(unittest.TestCase):
    def setUp(self):
        self.time = pd.Series(
            pd.date_range("2018-01-01 11:00:00", "2018-01-01 15:00:00", freq="H")
        )
        self.long = pd.DataFrame(
            {
                "TIME": pd.concat([self.time] * 6).values,
                "ID": pd.Series(["A", "B", "C"]).repeat(10).values,
                "TYPE": pd.Series(["X", "Y", "X", "Y", "X", "Y"]).repeat(5).values,
                "VALUE": range(30),  # 0 to 29
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        self.wide = pd.DataFrame(
            {
                "TIME": self.time,
                "A_X": [0, 1, 2, 3, 4],
                "A_Y": [5, 6, 7, 8, 9],
                "B_X": [10, 11, 12, 13, 14],
                "B_Y": [15, 16, 17, 18, 19],
                "C_X": [20, 21, 22, 23, 24],
                "C_Y": [25, 26, 27, 28, 29],
            },
            columns=["TIME", "A_X", "A_Y", "B_X", "B_Y", "C_X", "C_Y"],
        )

    def test_normal_to_wide(self):
        result = sam_format_to_wide(self.long)
        assert_frame_equal(result, self.wide)

    def test_normal_to_long(self):
        result = wide_to_sam_format(self.wide)
        assert_frame_equal(result, self.long)

    def test_alternate_colnames(self):

        wide2 = self.wide.copy()
        wide2 = wide2.rename(
            {
                "A_X": "A#X",
                "B_X": "B#X",
                "C_X": "C#X",
                "A_Y": "A#Y",
                "B_Y": "B#Y",
                "C_Y": "C#Y",
                "TIME": "datetime",
            },
            axis=1,
        )

        result1 = wide_to_sam_format(wide2, sep="#", timecol="datetime")
        assert_frame_equal(result1, self.long)

        expected = wide2.rename({"datetime": "TIME"}, axis=1)
        result2 = sam_format_to_wide(result1, sep="#")
        assert_frame_equal(result2, expected)

    def test_messedup_seps(self):

        wide2 = self.wide.copy()
        # Keep the _X the same, but the _Y are messed with
        wide2 = wide2.rename({"A_Y": "A_", "B_Y": "Y", "C_Y": "_Y"}, axis=1)

        expected = self.long.copy()
        # B_Y and C_Y are now missing id
        expected.loc[(expected["TYPE"] == "Y") & (expected["ID"] == "B"), "ID"] = "missingid"
        expected.loc[(expected["TYPE"] == "Y") & (expected["ID"] == "C"), "ID"] = "missingid"
        # A_Y is missing TYPE
        expected.loc[(expected["TYPE"] == "Y") & (expected["ID"] == "A"), "TYPE"] = ""

        result = wide_to_sam_format(wide2, idvalue="missingid")
        assert_frame_equal(result, expected)

    def test_missing_sep(self):
        result = wide_to_sam_format(self.wide, sep=None, idvalue="foo")

        expected = self.long.copy()
        expected["TYPE"] = expected["ID"] + "_" + expected["TYPE"]
        expected["ID"] = "foo"

        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
