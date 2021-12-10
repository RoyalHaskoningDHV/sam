import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from sam.feature_engineering import decompose_datetime
from sam.utils import FunctionTransformerWithNames
from sklearn.preprocessing import FunctionTransformer


class TestSklearnHelpers(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "TIME": pd.to_datetime(
                    ["2018-01-01 10:00", "2018-01-01 11:00:00", "2018-01-01 12:00:00"]
                ),
                "VALUE": [1, 2, 3],
            },
            columns=["TIME", "VALUE"],
        )

    def test_func_transformer(self):
        transformer = FunctionTransformerWithNames(
            decompose_datetime, kw_args={"components": ["hour", "minute"]}
        )

        original_transformer = FunctionTransformer(
            decompose_datetime,
            validate=False,
            kw_args={"components": ["hour", "minute"]},
        )

        newdata = transformer.fit_transform(self.data)
        expectednames = ["TIME", "VALUE", "TIME_hour", "TIME_minute"]
        resultnames = transformer.get_feature_names()
        self.assertEqual(expectednames, resultnames)

        expecteddata = original_transformer.fit_transform(self.data)
        assert_frame_equal(expecteddata, newdata)

    def test_functransformer_incorrect(self):
        arraydata = self.data.values
        transformer = FunctionTransformerWithNames(decompose_datetime)
        self.assertRaises(Exception, transformer.fit_transform, arraydata)


if __name__ == "__main__":
    unittest.main()
