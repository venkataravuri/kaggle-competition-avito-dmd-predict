import gc

import numpy as np
import pandas as pd
from time import time

import config
from utils import logging_utils, time_utils


def load_data():
    now = time_utils._timestamp()
    logname = "data_loader_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    t0 = time()
    logger.info('Loading raw data ...')
    train = pd.read_csv(config.TRAIN_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS)
    test = pd.read_csv(config.TEST_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS)
    logger.info('Loading raw data took: %s minutes' % round((time() - t0) / 60, 1))

    return train, test
