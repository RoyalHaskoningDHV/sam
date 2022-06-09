import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# Below are needed for setting up tests
from sam.preprocessing import normalize_timestamps


class TestCompleteTimestamps(unittest.TestCase):
    def test_normalize_timestamps(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        data_backup = data.copy()

        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        result = normalize_timestamps(data, "15min", start_time, end_time)

        # Values are matched to their first right side matching time,
        # so the first value is np.NaN
        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [np.nan, 1, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)
        # Make sure function has no side effects
        assert_frame_equal(data, data_backup)

    def test_normalize_with_timezone(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ).tz_localize("Asia/Qyzylorda"),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        data_backup = data.copy()

        # Test these in either string or tz-aware timestamp
        start_time = "2018/01/01 15:45:00"
        end_time = "2018/01/01 16:30:00"
        start_time2 = pd.Timestamp("2018/01/01 15:45:00", tz="Asia/Qyzylorda")
        end_time2 = pd.Timestamp("2018/01/01 16:30:00", tz="Asia/Qyzylorda")

        result = normalize_timestamps(data, "15min", start_time, end_time)
        result2 = normalize_timestamps(data, "15min", start_time2, end_time2)

        # Values are matched to their first left side matching time,
        # so the last value is np.NaN
        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ).tz_localize("Asia/Qyzylorda"),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [np.nan, 1, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        assert_frame_equal(result, output)
        assert_frame_equal(result2, output)
        # Make sure function has no side effects
        assert_frame_equal(data, data_backup)

    def test_fillna_method(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        # after ffill, the nan is filled in
        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 1, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        result = normalize_timestamps(data, "15min", start_time, end_time, fillna_method="bfill")
        assert_frame_equal(result, output, check_dtype=False)  # bfill changes to float

    def test_agg_method(self):
        # When using the sum, the sum of values 2 and 3 are taken within
        # the block 16:00 - 16:15 are taken

        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        # after ffill, the nan is filled in
        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [np.nan, 1, 5, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        result = normalize_timestamps(data, "15min", start_time, end_time, aggregate_method="sum")
        assert_frame_equal(result, output)

    def test_empty_start_end_time(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        result = normalize_timestamps(data, "15min", start_time="", end_time="")
        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)

    def test_multiple_ids(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": [1, 1, 2, 2],
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        start_time = pd.to_datetime("2018/01/01 16:00:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        result = normalize_timestamps(data, "15min", start_time, end_time)

        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": [1, 1, 1, 2, 2, 2],
                "TYPE": 2,
                "VALUE": [1.0, 2.0, np.NaN, 3.0, np.NaN, 4.0],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)

    def test_multiple_ids_fillna_method(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": [1, 1, 2, 2],
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        start_time = pd.to_datetime("2018/01/01 16:00:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        result = normalize_timestamps(data, "15min", start_time, end_time, fillna_method="ffill")

        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": [1, 1, 1, 2, 2, 2],
                "TYPE": 2,
                "VALUE": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)

    def test_multiple_types(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": [1, 1, 2, 2],
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        result = normalize_timestamps(data, "15min")

        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 16:30:00",
                    ]
                ),
                "ID": 1,
                "TYPE": [1, 1, 1, 2, 2, 2],
                "VALUE": [1, 2, np.NaN, 3, np.NaN, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)

    def test_round_method(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": [1, 1, 2, 2],
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )

        result = normalize_timestamps(data, "15min", round_method="floor")

        output = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                        "2018/01/01 15:45:00",
                        "2018/01/01 16:00:00",
                        "2018/01/01 16:15:00",
                    ]
                ),
                "ID": 1,
                "TYPE": [1, 1, 1, 2, 2, 2],
                "VALUE": [1, 2, np.NaN, 3, np.NaN, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        assert_frame_equal(result, output)

    def test_incorrect_input(self):
        data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    [
                        "2018/01/01 15:45:09",
                        "2018/01/01 16:03:09",
                        "2018/01/01 16:10:09",
                        "2018/01/01 16:22:09",
                    ]
                ),
                "ID": 1,
                "TYPE": 2,
                "VALUE": [1, 2, 3, 4],
            },
            columns=["TIME", "ID", "TYPE", "VALUE"],
        )
        start_time = pd.to_datetime("2018/01/01 15:45:00")
        end_time = pd.to_datetime("2018/01/01 16:30:00")

        # half uur is invalid timeunit
        self.assertRaises(ValueError, normalize_timestamps, data, "half uur", start_time, end_time)
        # wrong is not a time
        # integers are actually allowed, they are interpreted as UNIX time
        self.assertRaises(ValueError, normalize_timestamps, data, "15min", "wrong", end_time)

        data.columns = ["TIME", "ID", "TYPE", "SOMETHINGELSE"]
        self.assertRaises(
            Exception, normalize_timestamps, data, "15min", start_time, end_time, "sum"
        )

        self.assertRaises(
            Exception,
            normalize_timestamps,
            data,
            "15min",
            start_time,
            end_time,
            "unknown_fun",
            "",
        )
        self.assertRaises(
            Exception,
            normalize_timestamps,
            data,
            "15min",
            start_time,
            end_time,
            "",
            "unknown_fun",
        )

        self.assertRaises(Exception, normalize_timestamps, data, "15min", round_function="flotor")
        self.assertRaises(
            Exception,
            normalize_timestamps,
            data,
            "15min",
            fillna_method="supersmartfill",
        )
        self.assertRaises(Exception, normalize_timestamps, data.assign(ID=np.nan), "15min")
        self.assertRaises(Exception, normalize_timestamps, data.assign(TYPE=np.nan), "15min")
        self.assertRaises(Exception, normalize_timestamps, data.iloc[[]], "15min")


if __name__ == "__main__":
    unittest.main()
