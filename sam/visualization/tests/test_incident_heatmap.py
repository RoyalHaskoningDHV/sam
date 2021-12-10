import unittest

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from sam.visualization import plot_incident_heatmap


class TestIncidentHeatmap(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_incident_heatmap(self):
        range = pd.date_range("1/1/2011", periods=45, freq="D")
        ts = pd.DataFrame(
            {
                "incident": np.tile([0.1, 0.2, 0.3], 15),
                "id": np.tile(["A", "B", "C"], 15),
            },
            index=range,
            columns=["incident", "id"],
        )

        # Create the heatmap
        ax = plot_incident_heatmap(
            ts,
            resolution="W",
            annot=True,
            cmap="Reds",
            figsize=(15, 8),
            datefmt="%Y-%m-%d",
        )
        return ax.get_figure()

    @pytest.mark.mpl_image_compare(tolerance=30)
    def test_incident_heatmap_advanced(self):
        range = pd.date_range("1/1/2011", periods=45, freq="D")
        ts = pd.DataFrame(
            {
                "incident": np.tile([0.1, 0.2, 0.3], 15),
                "id": np.tile(["A", "B", "C"], 15),
            },
            index=range,
            columns=["incident", "id"],
        )

        # Create the heatmap
        pal = sns.light_palette("navy", reverse=True)
        ax = plot_incident_heatmap(
            ts,
            resolution="row",
            normalize=True,
            cmap=pal,
            datefmt="%Y-%m-%d",
            figsize=(15, 9),
        )
        return ax.get_figure()


if __name__ == "__main__":
    unittest.main()
