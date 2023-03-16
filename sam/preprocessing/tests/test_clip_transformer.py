import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from sam.preprocessing import ClipTransformer


class TestClipTransformer(unittest.TestCase):
    train_data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [3, 4, 5, 6, 7],
            "C": [1, 2, 3, 4, 5],
        }
    )
    test_input = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [3, -1, 5, 6, 10],
            "C": [1, 2, 0, 4, 10],
        }
    )
    test_output = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [3, 3, 5, 6, 7],
            "C": [1, 2, 1, 4, 5],
        }
    )
    test_output_fixed = pd.DataFrame(
        {
            "A": [2, 2, 3, 4, 4],
            "B": [3, 2, 4, 4, 4],
            "C": [2, 2, 2, 4, 4],
        }
    )

    def test_fit_transform(self):
        clipper = ClipTransformer()
        output = clipper.fit_transform(self.train_data)
        assert_frame_equal(output, self.train_data)

    def test_transform(self):
        clipper = ClipTransformer().fit(self.train_data)
        output = clipper.transform(self.test_input)
        assert_frame_equal(output, self.test_output)

    def test_transform_single_col(self):
        for column in ["A", "B", "C"]:
            clipper = ClipTransformer(cols=[column]).fit(self.train_data)
            output = clipper.transform(self.test_input)
            assert_series_equal(output[column], self.test_output[column])
            # check if other column have not changed
            assert_frame_equal(output.drop(column, axis=1), self.test_input.drop(column, axis=1))

    def test_min_max(self):
        clipper = ClipTransformer(min_value=2, max_value=4).fit(self.train_data)
        output = clipper.transform(self.test_input)
        assert_frame_equal(output, self.test_output_fixed)
