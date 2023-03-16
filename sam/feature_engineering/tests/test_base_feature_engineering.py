import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from sam.feature_engineering import BaseFeatureEngineer, FeatureEngineer, IdentityFeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]})
    X_out = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7], "C": [4, 6, 8, 10, 12]})

    def test_base_feature_engineer(self):
        with self.assertRaises(TypeError):
            BaseFeatureEngineer()

    def test_feature_engineer(self):
        def feature_engineer(X, y=None):
            X["C"] = X["A"] + X["B"]
            return X

        fe = FeatureEngineer(feature_engineer)
        self.assertIsInstance(fe, FeatureEngineer)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, self.X_out)

    def test_identity_feature_engineer(self):
        fe = IdentityFeatureEngineer()
        self.assertIsInstance(fe, IdentityFeatureEngineer)
        X_out = fe.fit_transform(self.X)
        assert_frame_equal(X_out, self.X)
