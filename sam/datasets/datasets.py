from pathlib import Path

import pandas as pd


PACKAGEDIR = Path(__file__).parent.absolute()


def load_rainbow_beach():
    """
    Loads the Rainbow Beach dataset (subset of the open Chicago Water dataset)

    Source:
    https://data.cityofchicago.org/Parks-Recreation/Beach-Water-Quality-Automated-Sensors/qmqz-2xku
    """
    file_path = PACKAGEDIR / "data/rainbow_beach.csv"
    return pd.read_csv(file_path, index_col=[0], parse_dates=[0])


def load_sewage_data():
    """
    Loads a sewage dataset, containing the discharge of multiple pumps and some weather data

    Source: Fake dataset by Royal HaskoningDHV
    """
    file_path = PACKAGEDIR / "data/sewage_data.csv"
    return pd.read_csv(file_path, index_col=[0], parse_dates=[0])
