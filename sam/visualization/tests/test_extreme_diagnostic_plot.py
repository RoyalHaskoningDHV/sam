import unittest

import numpy as np
import pandas as pd
import pytest
from sam.validation import RemoveExtremeValues
from sam.visualization import diagnostic_extreme_removal


class TestExtremeDiagnosticPlot(unittest.TestCase):
    def setUp(self):

        # create some random data
        np.random.seed(10)
        fs = [25, 100, 500]
        amps = [5, 2, 1]
        N = 1000
        x = np.arange(N)
        data = np.random.random(size=(N)) * 2
        for f, a in zip(fs, amps):
            data += np.sin(2 * np.pi * f * x / 8000) * a
        # data = np.random.random(size=(1000))
        # divide in train test set
        self.train_df = pd.DataFrame()
        self.train_df["values"] = data[:800]
        self.test_df = pd.DataFrame()
        self.test_df["values"] = data[:800]

        # with two clear outliers in train set
        self.train_df.iloc[250] *= 5
        self.train_df.iloc[500] *= 10

        # with one clear outlier in test set
        self.test_df.iloc[100] *= 3

        # now detect extremes
        cols_to_check = ["values"]
        self.REV = RemoveExtremeValues(cols=cols_to_check, rollingwindow=10, madthresh=10)

        _ = self.REV.fit(self.train_df)

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_extreme_values_removal_trainset(self):

        _ = self.REV.transform(self.train_df)
        fig = diagnostic_extreme_removal(self.REV, self.train_df, "values")
        return fig

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_extreme_values_removal_testset(self):

        _ = self.REV.transform(self.test_df)
        fig = diagnostic_extreme_removal(self.REV, self.test_df, "values")
        return fig


if __name__ == "__main__":
    unittest.main()
