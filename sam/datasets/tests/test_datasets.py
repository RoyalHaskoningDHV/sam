import unittest
from sam.datasets import load_rainbow_beach, load_sewage_data


class TestDatasets(unittest.TestCase):
    def test_rainbow_beach(self):
        df = load_rainbow_beach()
        print(df)
        self.assertEqual(df.index.name, "TIME")
        self.assertEqual(df.index.dtype, "datetime64[ns]")
        self.assertListEqual(
            df.columns.tolist(),
            [
                "battery_life",
                "transducer_depth",
                "turbidity",
                "water_temperature",
                "wave_height",
                "wave_period",
            ],
        )

    def test_sewage_data(self):
        df = load_sewage_data()
        print(df)
        self.assertEqual(df.index.name, "TIME")
        self.assertEqual(df.index.dtype, "datetime64[ns]")
        self.assertListEqual(
            df.columns.tolist(),
            [
                "Discharge_Hoofdgemaal",
                "Discharge_Sportlaan",
                "Discharge_Emmastraat",
                "Discharge_Kerkstraat",
                "Precipitation",
                "Temperature",
            ],
        )
