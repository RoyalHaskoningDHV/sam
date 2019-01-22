import unittest
import pytest
from sam.visualization import make_incident_heatmap
import pandas as pd
import numpy as np
import matplotlib


class TestIncidentHeatmap(unittest.TestCase):

    @pytest.mark.mpl_image_compare
    def test_incident_heatmap(self):
        range = pd.date_range('1/1/2011', periods=45, freq='D')
        ts = pd.DataFrame({'incident': np.tile([0.1, 0.2, 0.3], 15),
                           'id': np.tile(['A', 'B', 'C'], 15)},
                          index=range, columns=['incident', 'id'])

        # Create the heatmap
        ax = make_incident_heatmap(ts, resolution='W', annot=True, cmap='Reds')
        return ax.get_figure()

if __name__ == '__main__':
    unittest.main()
