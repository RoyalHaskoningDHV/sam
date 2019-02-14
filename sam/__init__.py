import logging
import datetime
from os.path import isdir

# Only log if the directory exists, stops erros on unit tests
if isdir("logs"):
    # We take only the message from the sam package, not matplotlib etc.
    logger = logging.getLogger('sam')
    fh = logging.FileHandler('logs/sam_' + datetime.datetime.now().strftime('%Y-%m-%d') +
                             '.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

__all__ = ['data_sources', 'feature_engineering', 'feature_extraction', 'feature_selection',
           'metrics', 'preprocessing', 'train_models', 'utils', 'visualization']
