from typing import Dict, Sequence

import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import Callback
from sam.metrics.r2_calculation import train_r2


class R2Evaluation(Callback):
    def __init__(
        self,
        all_data: Dict[str, np.ndarray],
        prediction_cols: list,
        predict_ahead: Sequence[int],
    ):
        """
        Custom keras callback that computes r2 compared to the training mean.
        Computing R2 at every batch and then averaging biases r2 estimates.
        Implementing r2 as a metric is therefore not valid, as this is evaluated every batch.
        We therefore implemented it as a callback, which is only evaluated at the end of each
        epoch.

        NOTE: this should only be used with MLPTimeseriesRegressor models, not any custom keras
        model.
        NOTE-2: this function returns r2 with the keras_model.predict function. This means that
            if values are differences in MLPTimeseriesRegressor, it returns r2 for the differenced
            values. This can deviate from r2 computed over un-differenced values.

        Parameters
        ----------
        all_data: dict
            Dictionary that should include X_train and y_train at least. If validation set is
            present should also include X_val and y_val. The training sets individually should
            be numpy arrays, and should be the same as are input to the model for training.
        prediction_cols: list
            List of columns that accompany a model.predict call
        predict_ahead: integer
            Number of timesteps ahead

        """
        self.all_data = all_data
        self.prediction_cols = prediction_cols
        self.predict_ahead = predict_ahead

    def on_epoch_end(self, epoch, logs=None):
        """
        Computes r2 compared to the training mean for each predict_ahead, calculated the
        average and prints the result
        """
        if logs is None:
            logs = {}

        val = "X_val" in self.all_data.keys()

        y_hat_train = pd.DataFrame(
            data=self.model.predict(self.all_data["X_train"]),
            index=self.all_data["X_train"].index,
            columns=self.prediction_cols,
        )

        if val:
            y_hat_val = pd.DataFrame(
                data=self.model.predict(self.all_data["X_val"]),
                index=self.all_data["X_val"].index,
                columns=self.prediction_cols,
            )

        r2s = []
        r2s_val = []
        for p in self.predict_ahead:
            if len(self.predict_ahead) > 1:
                # only add the predict ahead if needed
                thiscol = "_".join(self.all_data["y_train"].columns[0].split("_")[:-2])
                thiscol += "_lead_%d" % p
            else:
                thiscol = self.all_data["y_train"].columns[0]

            these_y_train = self.all_data["y_train"].loc[:, thiscol].values
            these_y_hat_train = y_hat_train["predict_lead_%d_mean" % p].values
            train_mean = these_y_train.mean()

            if val:
                these_y_val = self.all_data["y_val"].loc[:, thiscol].values
                these_y_hat_val = y_hat_val["predict_lead_%d_mean" % p].values

            r2s.append(train_r2(these_y_train, these_y_hat_train, train_mean))

            if val:
                r2s_val.append(train_r2(these_y_val, these_y_hat_val, train_mean))

        r2 = np.mean(r2s)
        logs["r2"] = r2
        if val:
            r2_val = np.mean(r2s_val)
            logs["val_r2"] = r2_val

        if val:
            print("r2: {:.6f} - val_r2: {:.6f}".format(r2, r2_val))
        else:
            print("r2: {:.6f}".format(r2))
