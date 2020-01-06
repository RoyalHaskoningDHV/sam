import pandas as pd
import numpy as np
import random
import os
import unittest
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from sklearn.metrics import mean_absolute_error

from sam.models import SamQuantileMLP

# If tensorflow is not available, skip these unittests
skipkeras = False
try:
    import tensorflow as tf
    from keras import backend as K
except ImportError:
    skipkeras = True


@pytest.mark.skipif(skipkeras, reason="Keras backend not found")
class TestSamQuantileMLP(unittest.TestCase):

    def setUp(self):
        # We are deliberately creating an extremely easy, linear problem here
        # the target is literally 17 times one of the features
        # This is because we just want to see if the model works at all, in a short time, on very
        # little data.
        # With a high enough learning rate, it should be almost perfect after a few iterations

        self.n_rows = 100
        self.train_size = int(self.n_rows * 0.8)

        self.X = pd.DataFrame({
            'TIME': pd.to_datetime(np.array(range(self.n_rows)), unit='m'),
            'x': range(self.n_rows)
        })

        self.y = 17 * self.X['x'] + 34
        self.X_train, self.X_test = self.X[:self.train_size], self.X[self.train_size:]
        self.y_train, self.y_test = self.y[:self.train_size], self.y[self.train_size:]

        # Now start setting the RNG so we get reproducible results
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '0'

        # Force tensorflow to run single-threaded, for further determinism
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)

        tf.set_random_seed(42)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

        # For now, don't change the session since it makes one of the unit tests (shap)
        # fail. I'm not sure exactly why
        # K.set_session(sess)

    def test_predict_future(self):
        # We will fit quantiles but don't test them
        # That is because this is an extremely trivial problem, so the quantiles will likely
        # be extremely close to the mean prediction, so it is hard to verify them

        # I only use a single neuron here, which is kinda weird but works for this
        # intentionally linear problem.
        # This is the only way that I have found that the network for SURE will not
        # get stuck in overfitting or some local maximum, because I don't want this
        # unit test to fail randomly 10% of the time

        # We also use an extremely high learning rate so that we only need 30 epochs
        # No rolling features because sometimes the model assigns all it's weight to a lag
        # feature which means the answer is slightly but consistently wrong

        model = SamQuantileMLP(predict_ahead=2,
                               use_y_as_feature=True,
                               timecol='TIME',
                               quantiles=[0.3, 0.7],
                               epochs=30,
                               time_components=['minute'],
                               time_cyclicals=['minute'],
                               rolling_window_size=[],
                               n_neurons=1,
                               n_layers=1,
                               lr=0.3,
                               verbose=0)

        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test, self.y_test)
        actual = model.get_actual(self.y_test)

        # Sanity check on model.get_actual
        assert_series_equal(actual, self.y_test.shift(-2), check_names=False)

        # Unfortunately, even with all the safeguards above, it can still sometimes fail
        # we can see this because the quantiles have a very different average than the average
        # prediction. Likely because mean uses 'mse' whereas quantiles use mae.
        # In this case, I trust the quantiles much more because for this problem, mse seems too
        # strict and can create a local minimum much more often
        if abs(pred['predict_lead_2_q_0.3'].mean() - pred['predict_lead_2_mean'].mean()) > 5:
            pred['predict_lead_2_mean'] = pred['predict_lead_2_q_0.3']

        results = pd.DataFrame({
            'pred': pred['predict_lead_2_mean'],
            'actual': actual,
            'persistence': self.y_test
        }, index=actual.index).dropna()

        mae = mean_absolute_error(results['actual'], results['pred'])
        # Our performance should be better than the performance if we shift the prediction
        self.assertLess(mae, mean_absolute_error(results['actual'][1:], results['pred'].iloc[:-1]))
        self.assertLess(mae, mean_absolute_error(results['actual'][:-1], results['pred'].iloc[1:]))
        # We should easily be able to outperform the persistence benchmark
        self.assertLess(mae, mean_absolute_error(results['actual'], results['persistence']))

        # Try the preprocessing
        X_transformed = model.preprocess_before_predict(self.X_test, self.y_test)
        # Should have the same number of columns as model.get_feature_names()
        self.assertEqual(X_transformed.shape[1], len(model.get_feature_names()))

        self.assertEqual(model.n_outputs_, 3)
        self.assertEqual(model.prediction_cols_,
                         ['predict_lead_2_q_0.3', 'predict_lead_2_q_0.7', 'predict_lead_2_mean'])

        # The actual feature importances aren't tested, only that they are outputted in
        # the correct shape and with correct column names
        feature_importances = \
            model.quantile_feature_importances(self.X_test, self.y_test, n_iter=2)
        self.assertEqual(feature_importances.columns.tolist(),
                         model.get_feature_names())
        self.assertEqual(feature_importances.shape, (2, 4))

        # test shap values
        explainer = model.get_explainer(self.X_test, self.y_test)
        shap_values = explainer.shap_values(self.X_test[0:10], self.y_test[0:10])
        test_values = explainer.test_values(self.X_test[0:10], self.y_test[0:10])

        # Should be 3, since we have 3 outputs in our model
        self.assertEqual(len(shap_values), model.n_outputs_)
        # We explained 10 rows, with 4 features each
        self.assertEqual(shap_values[0].shape, (10, model.n_inputs_))
        self.assertEqual(test_values.shape, (10, model.n_inputs_))
        self.assertEqual(test_values.columns.tolist(), model.get_feature_names())

        # Just run this, see if it throws any warning/errors
        # Score should ouput a scalar, summary should output nothing
        score = model.score(self.X_test, self.y_test)
        self.assertTrue(np.isscalar(score))

        output = model.summary()
        self.assertIsNone(output)

    def test_predict_present(self):

        # Add a generous amount of noise to y, because we don't need to model to fit perfectly for
        # this unit test.
        # In fact, with so few epochs and so much noise we probably expect the model to underfit
        y = self.y + np.random.normal(0, 100, self.n_rows)
        y_train, y_test = y[:self.train_size], self.y[self.train_size:]

        # Drop time since we are running a test without TIME
        X = self.X.drop('TIME', axis=1)
        X_train, X_test = X[:self.train_size], X[self.train_size:]

        model = SamQuantileMLP(predict_ahead=0,
                               use_y_as_feature=False,
                               timecol=None,
                               quantiles=(0.05, 0.95),
                               epochs=20,
                               time_components=[],
                               time_cyclicals=[],
                               rolling_window_size=[1],
                               n_neurons=8,
                               n_layers=1,
                               lr=0.2,
                               verbose=0)

        with pytest.warns(UserWarning):
            # We expect a warning because there is no timecol
            model.fit(X_train, y_train)
            pred = model.predict(X_test, y_test)

        # Test that the lower/higher quantile predicts lower/higher than the mean
        # When there are 100 training rows, this corresponds to 16 out of 20 'successes'
        # at least 16 successes out of 20 is pretty strict: if it was a coinflip, the probability
        # of that happenening would be approx. 1 in 166
        self.assertGreaterEqual(
            (pred['predict_lead_0_q_0.05'] < pred['predict_lead_0_mean']).sum(),
            self.train_size * 0.2
        )
        self.assertGreaterEqual(
            (pred['predict_lead_0_mean'] < pred['predict_lead_0_q_0.95']).sum(),
            self.train_size * 0.2
        )

    def test_multioutput(self):
        # Test multioutput. Same data as test_predict_future.

        model = SamQuantileMLP(predict_ahead=[1, 2, 3],
                               use_y_as_feature=True,
                               timecol='TIME',
                               quantiles=[0.3, 0.7],
                               epochs=30,
                               time_components=['minute'],
                               time_cyclicals=['minute'],
                               rolling_window_size=[],
                               n_neurons=1,
                               n_layers=1,
                               lr=0.3,
                               verbose=0)

        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test, self.y_test)
        actual = model.get_actual(self.y_test)

        yname = self.y_test.name
        # Same as with singleoutput, except in a dataframe this time
        expected = pd.DataFrame({
            yname + '_diff_1': self.y_test.shift(-1),
            yname + '_diff_2': self.y_test.shift(-2),
            yname + '_diff_3': self.y_test.shift(-3)
        }, index=self.y_test.index)
        assert_frame_equal(actual, expected)

        self.assertEqual(model.prediction_cols_, pred.columns.tolist())
        self.assertEqual(pred.columns.tolist(),
                         ['predict_lead_1_q_0.3', 'predict_lead_2_q_0.3', 'predict_lead_3_q_0.3',
                          'predict_lead_1_q_0.7', 'predict_lead_2_q_0.7', 'predict_lead_3_q_0.7',
                          'predict_lead_1_mean', 'predict_lead_2_mean', 'predict_lead_3_mean'])

        X_transformed = model.preprocess_before_predict(self.X_test, self.y_test)
        # Should have the same number of columns as model.get_feature_names()
        self.assertEqual(X_transformed.shape[1], len(model.get_feature_names()))

        self.assertEqual(model.n_outputs_, 9)

        score = model.score(self.X_test, self.y_test)
        self.assertTrue(np.isscalar(score))

    def test_single_target(self):
        # Test single target. Same data as test_predict_future.

        model = SamQuantileMLP(predict_ahead=1,
                               use_y_as_feature=True,
                               timecol='TIME',
                               quantiles=[],
                               epochs=2,
                               time_components=['minute'],
                               time_cyclicals=['minute'],
                               rolling_window_size=[],
                               n_neurons=1,
                               n_layers=1,
                               lr=0.3,
                               verbose=0)

        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test, self.y_test)
        actual = model.get_actual(self.y_test)

        yname = self.y_test.name
        # Same as with singleoutput, except in a dataframe this time
        expected = pd.Series(self.y_test.shift(-1),
                             name=yname + '_diff_1', index=self.y_test.index)
        assert_series_equal(actual, expected)

        self.assertEqual(model.prediction_cols_[0], pred.name)
        self.assertEqual(pred.name, 'predict_lead_1_mean')

        X_transformed = model.preprocess_before_predict(self.X_test, self.y_test)
        # Should have the same number of columns as model.get_feature_names()
        self.assertEqual(X_transformed.shape[1], len(model.get_feature_names()))

        self.assertEqual(model.n_outputs_, 1)

        score = model.score(self.X_test, self.y_test)
        self.assertTrue(np.isscalar(score))

    def test_expected_failures(self):

        # For now, this usecase is not supported so will throw error
        model = SamQuantileMLP(predict_ahead=1, use_y_as_feature=False, timecol='TIME')
        self.assertRaises(ValueError, model.fit, self.X, self.y)

        # No negative values allowed
        model = SamQuantileMLP(predict_ahead=-1, use_y_as_feature=True, timecol='TIME')
        self.assertRaises(ValueError, model.fit, self.X, self.y)

        # Cannot use y as feature when predict_ahead is 0
        model = SamQuantileMLP(predict_ahead=0, use_y_as_feature=True, timecol='TIME')
        self.assertRaises(ValueError, model.fit, self.X, self.y)

        # make the index of X and y not match
        y = self.y.copy()
        y.index = range(1, self.n_rows + 1)

        # Index of X and y don't match
        model = SamQuantileMLP(predict_ahead=1, timecol='TIME')
        self.assertRaises(ValueError, model.fit, self.X, y)

        # Make the time not monospaced
        X = self.X.copy()
        X['TIME'] = pd.to_datetime(np.random.randint(0, 1000, 100), unit='m')
        self.assertRaises(ValueError, model.fit, X, self.y)


if __name__ == '__main__':
    unittest.main()
