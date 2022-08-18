import configparser
import datetime
import logging
import re
import warnings
from os.path import isdir


warnings.filterwarnings(
    "always", category=DeprecationWarning, module=r"^{0}\.".format(re.escape(__name__))
)

# Only log if the directory exists, stops errors on unit tests
if isdir("logs"):
    # We take only the message from the sam package, not matplotlib etc.
    logger = logging.getLogger("sam")
    fh = logging.FileHandler("logs/sam_" + datetime.datetime.now().strftime("%Y-%m-%d") + ".log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read(".config")

__all__ = [
    "data_sources",
    "datasets",
    "exploration",
    "feature_engineering",
    "logging_functions",
    "metrics",
    "models",
    "preprocessing",
    "utils",
    "validation",
    "visualization",
]
