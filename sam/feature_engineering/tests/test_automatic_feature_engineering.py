import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sam.feature_engineering import AutomaticRollingEngineering
from sam.feature_engineering.automatic_rolling_engineering import (
    INPUT_VALIDATION_ERR_MESSAGE,
)
from sklearn.model_selection import train_test_split

TEST_FOLDER = Path(__file__).parent.absolute()


class TestAutomaticRollingEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(10)

        # this is actual data from sam's read_knmi, using the following command:
        # read_knmi('2019-03-11', '2018-04-11', variables = ['T', 'Q'])
        data = pd.read_parquet(
            TEST_FOLDER / "test_data" / "20190311_till_20190411_knmi_data_T_Q.parquet"
        )

        # let's predict temperature 12 values ahead
        target = "T"
        fut = 12
        self.y = data[target].shift(-fut).iloc[:-fut]
        self.X = data.iloc[:-fut]

    def runAutomaticRollingEngineer(self):
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            self.X, self.y, shuffle=False
        )

        self.ARE = AutomaticRollingEngineering(
            window_sizes=[[8]], rolling_types=["lag"], n_iter_per_param=1, cv=2
        ).fit(self.X_train, y_train)

        self.r2_base, self.r2_rollings, _, _ = self.ARE.compute_diagnostics(
            self.X_train, self.X_test, y_train, y_test
        )

        self.X_train_rolling = self.ARE.transform(self.X_train)
        self.X_test_rolling = self.ARE.transform(self.X_test)

        # also fit second one with time features
        self.ARE2 = AutomaticRollingEngineering(
            window_sizes=[[8]],
            rolling_types=["lag"],
            n_iter_per_param=1,
            onehots=["weekday"],
            cyclicals=["secondofday"],
            cv=2,
        ).fit(self.X_train, y_train)

    def test_input_validation_datetimeindex(self):
        """Should raise AssertionError if index is not a DatetimeIndex"""
        self.X = self.X.reset_index(drop=True)

        with self.assertRaises(ValueError) as cm:
            self.runAutomaticRollingEngineer()

        self.assertEqual(cm.exception.args[0], INPUT_VALIDATION_ERR_MESSAGE)

    def test_input_validation_linear(self):
        """Should raise AssertionError if DatetimeIndex is not linear"""
        index = self.X.index.tolist()
        index[1] = index[0]
        self.X = self.X.reset_index(drop=True)
        self.X.index = index

        with self.assertRaises(ValueError) as cm:
            self.runAutomaticRollingEngineer()

        self.assertEqual(cm.exception.args[0], INPUT_VALIDATION_ERR_MESSAGE)

    def test_r2s(self):
        self.runAutomaticRollingEngineer()
        assert_almost_equal(self.r2_base, -1.1744610463988145)
        assert_almost_equal(self.r2_rollings, -0.9671777894504794)

    def test_column_names(self):
        self.runAutomaticRollingEngineer()
        assert_array_equal(self.X_train_rolling.columns, ["T", "T#lag_8", "Q", "Q#lag_8"])
        assert_array_equal(self.X_test_rolling.columns, ["T", "T#lag_8", "Q", "Q#lag_8"])

    def test_feature_importances(self):
        self.runAutomaticRollingEngineer()
        assert_array_almost_equal(
            self.ARE.feature_importances_["coefficients"].values,
            [0.1326634, 0.02229596, -0.13034243, -0.15520309],
        )

    def test_output_indices(self):
        self.runAutomaticRollingEngineer()
        assert_array_equal(self.X_train.index, self.X_train_rolling.index)
        assert_array_equal(self.X_test.index, self.X_test_rolling.index)

    def test_feature_names(self):
        self.runAutomaticRollingEngineer()
        assert_array_equal(
            self.ARE.feature_importances_.feature_name.unique(),
            ["T", "Q#lag_8", "T#lag_8", "Q"],
        )

    def test_feature_names_with_timefeatures(self):
        self.runAutomaticRollingEngineer()
        assert_array_equal(
            self.ARE2.feature_importances_.feature_name.unique(),
            [
                "TIME_weekday_0",
                "TIME_weekday_1",
                "TIME_weekday_6",
                "TIME_secondofday_cos",
                "T",
                "Q#lag_8",
                "Q",
                "T#lag_8",
                "TIME_secondofday_sin",
                "TIME_weekday_2",
                "TIME_weekday_5",
                "TIME_weekday_4",
                "TIME_weekday_3",
            ],
        )


if __name__ == "__main__":
    unittest.main()
