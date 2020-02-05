import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

import pandas as pd
import numpy as np
from sam.feature_engineering import AutomaticRollingEngineering
from sklearn.model_selection import train_test_split


class TestAutomaticRollingEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        np.random.seed(10)

        # this is actual data from sam's read_knmi, using the following command:
        # read_knmi('2018-01-01', '2018-01-03', variables = ['T', 'Q'])
        data = pd.DataFrame()
        data['T'] = [87., 85., 71., 78., 80., 75., 69., 65., 62., 66., 71., 74., 70.,
                     75., 75., 76., 64., 61., 60., 56., 54., 58., 61., 61., 53., 52.,
                     48., 54., 58., 61., 62., 56., 62., 61., 68., 69., 69., 70., 67.,
                     63., 60., 63., 63., 59., 62., 70., 79., 87., 89.]
        data['Q'] = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5., 38., 63., 58.,
                     35., 18.,  7.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                     0.,  0.,  0.,  0.,  0.,  0.,  0.,  8., 31., 53., 34., 26., 13.,
                     8.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]

        time1 = '2019/03/11 00:00:00'
        time2 = '2019/04/11 00:00:00'
        data.index = pd.date_range(time1, time2, periods=data.shape[0])

        # let's predict temperature 12 values ahead
        target = 'T'
        fut = 12
        y = data[target].shift(-fut).iloc[:-fut]
        X = data.iloc[:-fut]

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

        self.ARE = AutomaticRollingEngineering(
            window_sizes=[[8]],
            rolling_types=['lag'],
            n_iter_per_param=1,
            cv=2).fit(self.X_train, y_train)

        self.r2_base, self.r2_rollings, base_model, roll_model = \
            self.ARE.compute_diagnostics(self.X_train, self.X_test, y_train, y_test)

        self.X_train_rolling = self.ARE.transform(self.X_train)
        self.X_test_rolling = self.ARE.transform(self.X_test)

        # also fit second one with time features
        self.ARE2 = AutomaticRollingEngineering(
            window_sizes=[[8]],
            rolling_types=['lag'],
            n_iter_per_param=1,
            onehots=['weekday'],
            cyclicals=['secondofday'],
            cv=2).fit(self.X_train, y_train)

    def test_r2s(self):
        assert_almost_equal(self.r2_base, -1.1744610463988145)
        assert_almost_equal(self.r2_rollings, -0.9671777894504794)

    def test_column_names(self):
        assert_array_equal(self.X_train_rolling.columns, ['T', 'T#lag_8', 'Q', 'Q#lag_8'])
        assert_array_equal(self.X_test_rolling.columns, ['T', 'T#lag_8', 'Q', 'Q#lag_8'])

    def test_feature_importances(self):
        assert_array_almost_equal(
            self.ARE.feature_importances_['coefficients'].values,
            [0.1326634,  0.02229596, -0.13034243, -0.15520309])

    def test_output_indices(self):
        assert_array_equal(self.X_train.index, self.X_train_rolling.index)
        assert_array_equal(self.X_test.index, self.X_test_rolling.index)

    def test_feature_names(self):
        assert_array_equal(
            self.ARE.feature_importances_.feature_name.unique(),
            ['T', 'Q#lag_8', 'T#lag_8', 'Q'])

    def test_feature_names_with_timefeatures(self):
        assert_array_equal(
            self.ARE2.feature_importances_.feature_name.unique(),
            ['TIME_weekday_0', 'TIME_weekday_1', 'TIME_weekday_6',
             'TIME_secondofday_cos', 'T', 'Q#lag_8', 'Q', 'T#lag_8',
             'TIME_secondofday_sin', 'TIME_weekday_2', 'TIME_weekday_5',
             'TIME_weekday_4', 'TIME_weekday_3'])


if __name__ == '__main__':
    unittest.main()
