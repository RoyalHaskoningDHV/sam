# This test checks if all deprecated API's still work (with deprecation warning)
# Of course, once the deprecated API's are removed, these tests should also be removed
import unittest
import numpy as np
import pandas as pd

from sam.data_sources import create_synthetic_timeseries
from sam.preprocessing import complete_timestamps
from sam.feature_engineering import build_timefeatures
from sam.feature_selection import retrieve_top_n_correlations, retrieve_top_score_correlations, \
    create_lag_correlation
from sam.train_models import find_outlier_curves, create_outlier_information
from sam.utils import MongoWrapper, average_winter_time, label_dst, unit_to_seconds
from sam.visualization import make_precision_recall_curve, make_threshold_plot, \
    make_incident_heatmap


class TestDeprecatedAPIS(unittest.TestCase):

    def test_depr_create_synthetic_timeseries(self):
        dates = pd.date_range('2015-01-01', '2016-01-01', freq='6H').to_series()

        with self.assertWarns(DeprecationWarning):
            create_synthetic_timeseries(dates,
                                        monthly=5, daily=1, hourly=0.0,
                                        monthnoise=('normal', 0.01), daynoise=('normal', 0.01),
                                        noise={'normal': 0.1}, minmax_values=(5, 25),
                                        cutoff_values=None, random_missing=0.12)

    def test_depr_complete_timestamps(self):
        from datetime import datetime
        with self.assertWarns(DeprecationWarning):
            df = pd.DataFrame({'TIME': [datetime(2018, 6, 9, 11, 13), datetime(2018, 6, 9, 11, 34),
                                        datetime(2018, 6, 9, 11, 44), datetime(2018, 6, 9, 11, 4)],
                               'ID': "SENSOR", 'VALUE': [1, 20, 3, 20]})
            complete_timestamps(df, freq="15 min", end_time="2018-06-09 12:15:00",
                                aggregate_method="median", fillna_method=None)

    def test_depr_build_timefeatures(self):
        with self.assertWarns(DeprecationWarning):
            build_timefeatures("28-12-2018", "01-01-2019", freq="11 H")

    def test_depr_top_correlations(self):
        from sam.feature_engineering import BuildRollingFeatures
        goal_feature = 'DEBIET_TOTAAL#lag_0'
        df = pd.DataFrame({'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           'DEBIET_A': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3],
                           'DEBIET_B': [3, 1, 2, 3, 3, 6, 4, 1, 3, 3, 1, 5]})
        df['DEBIET_TOTAAL'] = df['DEBIET_A'] + df['DEBIET_B']

        with self.assertWarns(DeprecationWarning):
            df_copy = df.rename({'DEBIET_A': 'DEBIET#A', 'DEBIET_B': 'DEBIET#B',
                                 'DEBIET_TOTAAL': 'DEBIET#TOTAAL'}, axis=1)
            create_lag_correlation(df_copy, 'DEBIET#TOTAAL', lag=11)

        RollingFeatures = BuildRollingFeatures(rolling_type='lag',
                                               window_size=np.arange(10), lookback=0,
                                               keep_original=False)
        res = RollingFeatures.fit_transform(df)

        with self.assertWarns(DeprecationWarning):
            retrieve_top_n_correlations(res, goal_feature, n=2, grouped=True, sep='#')

        with self.assertWarns(DeprecationWarning):
            retrieve_top_score_correlations(res, goal_feature, score=0.8)

    def test_depr_find_outlier_curve(self):
        data = pd.DataFrame({'ACTUAL': [0.3, np.nan, 0.3, np.nan, 0.3, 0.5, np.nan, 0.7],
                             'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4})

        with self.assertWarns(DeprecationWarning):
            find_outlier_curves(data)

    def test_depr_find_outlier_curve_information(self):
        data = pd.DataFrame({'TIME': range(1547477436, 1547477436+3),  # unix timestamps
                             'ACTUAL': [0.3, 0.5, 0.7],
                             'PREDICT_HIGH': 0.6, 'PREDICT_LOW': 0.4, 'PREDICT': 0.5})
        with self.assertWarns(DeprecationWarning):
            create_outlier_information(data)

    def test_depr_mongowrapper(self):
        with self.assertWarns(DeprecationWarning):
            MongoWrapper('test', 'test')

    def test_depr_average_winter_time(self):
        daterange = pd.date_range('2019-10-27 01:45:00', '2019-10-27 03:00:00', freq='15min')
        test_df = pd.DataFrame({"TIME": daterange.values[[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]],
                                "VALUE": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])})
        with self.assertWarns(DeprecationWarning):
            average_winter_time(test_df)

    def test_depr_label_dst(self):
        daterange = pd.date_range('2019-10-27 01:00:00', '2019-10-27 03:00:00', freq='15min')
        with self.assertWarns(DeprecationWarning):
            label_dst(pd.Series(daterange))

    def test_depr_unit_to_seconds(self):
        with self.assertWarns(DeprecationWarning):
            unit_to_seconds("week")

    def test_depr_make_precision_recall_curve(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.3])
        with self.assertWarns(DeprecationWarning):
            make_precision_recall_curve(y_true, y_scores)

    def test_depr_make_threshold_plot(self):
        with self.assertWarns(DeprecationWarning):
            make_threshold_plot([0, 1, 0, 1, 1, 0], [.2, .3, .4, .5, .9, .1])

    def test_make_incident_heatmap(self):
        rng = pd.date_range('1/1/2011', periods=10, freq='D')
        ts = pd.DataFrame({'values': np.random.randn(len(rng)),
                           'id': np.random.choice(['A', 'B', 'C'], len(rng))},
                          index=rng, columns=['values', 'id'])
        ts['incident'] = 0
        ts.loc[ts['values'] > .5, 'incident'] = 1
        with self.assertWarns(DeprecationWarning):
            make_incident_heatmap(ts, resolution='W', annot=True,
                                  cmap='Reds', datefmt="%Y, week %W")


if __name__ == '__main__':
    unittest.main()
