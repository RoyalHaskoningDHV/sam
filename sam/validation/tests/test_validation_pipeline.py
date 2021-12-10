import unittest

import numpy as np
import pandas as pd
from sam.validation import create_validation_pipe

from .numeric_assertions import NumericAssertions


class TestValidationPipeline(unittest.TestCase, NumericAssertions):
    def test_validation_pipeline(self):

        # create some data
        np.random.seed(10)
        base = np.random.randn(100)
        X_train = pd.DataFrame(np.tile(base, (3, 3)).T, columns=["1", "2", "3"])
        X_test = pd.DataFrame(np.tile(base, (3, 1)).T, columns=["1", "2", "3"])
        y_test = pd.Series(base, name="target")
        y_train = pd.Series(np.tile(base, 3).T, name="target")

        # add outliers to y_train and y_test:
        y_train.iloc[[5, 10, 61]] *= 30
        y_test.iloc[[5, 10, 61]] *= 30
        # add flatlines to y_train and y_test:
        y_test.iloc[20:40] = 1
        y_train.iloc[20:100] = 1

        # setup pipeline
        pipe = create_validation_pipe(
            cols=list(X_train.columns) + ["target"],
            rollingwindow=5,
            impute_method="iterative",
        )

        # put data together
        train_data = X_train.join(y_train)
        test_data = X_test.join(y_test)

        # now fit the pipeline on the train data and transform both train and test
        train_data = pd.DataFrame(
            pipe.fit_transform(train_data),
            columns=train_data.columns,
            index=train_data.index,
        )
        test_data = pd.DataFrame(
            pipe.transform(test_data), columns=test_data.columns, index=test_data.index
        )

        # now split in X and y again
        y_test = test_data["target"]
        y_train = train_data["target"]
        X_train = train_data.drop("target", axis=1)
        X_test = test_data.drop("target", axis=1)

        # test whether there are no more nans in any dataset
        self.assertAllNotNaN(y_test)
        self.assertAllNotNaN(y_train)
        self.assertAllNotNaN(X_train)
        self.assertAllNotNaN(X_test)


if __name__ == "__main__":
    unittest.main()
