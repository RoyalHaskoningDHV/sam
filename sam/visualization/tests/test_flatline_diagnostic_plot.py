import unittest

import pandas as pd
import pytest
from sam.validation import RemoveFlatlines
from sam.visualization import diagnostic_flatline_removal


class TestFlatlineDiagnosticPlot(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_flatline_removal(self):

        # create some random data
        data = [1, 2, 6, 3, 4, 4, 4, 3, 6, 7, 7, 2, 2]
        # with one clear outlier
        test_df = pd.DataFrame()
        test_df["values"] = data
        # now detect extremes
        cols_to_check = ["values"]
        RF = RemoveFlatlines(cols=cols_to_check, window=2)
        _ = RF.fit_transform(test_df)

        fig = diagnostic_flatline_removal(RF, test_df, "values")

        return fig


if __name__ == "__main__":
    unittest.main()
