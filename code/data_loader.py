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
    logger.info('Loading raw train & test data ...')
    train = pd.read_csv(config.TRAIN_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS,
                        dtype=config.DTYPES, encoding='utf8')
    test = pd.read_csv(config.TEST_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS,
                       dtype=config.DTYPES, encoding='utf8')
    logger.info('Loading raw train & test data took: %s minutes' % round((time() - t0) / 60, 1))

    return train, test

def load_train_data():
    now = time_utils._timestamp()
    logname = "data_loader_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    t0 = time()
    logger.info('Loading raw train data ...')
    train = pd.read_csv(config.TRAIN_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS,
                        dtype=config.DTYPES, encoding='utf8')
    logger.info('Loading raw train data took: %s minutes' % round((time() - t0) / 60, 1))

    return train


def load_test_data():
    now = time_utils._timestamp()
    logname = "data_loader_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    t0 = time()
    logger.info('Loading raw test data ...')
    test = pd.read_csv(config.TEST_DATA, parse_dates=['activation_date'], nrows=config.RAW_DATA_ROWS,
                        dtype=config.DTYPES, encoding='utf8')
    logger.info('Loading raw test data took: %s minutes' % round((time() - t0) / 60, 1))

    return test


def load_periods_data():
    now = time_utils._timestamp()
    logname = "data_loader_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    t0 = time()
    logger.info('Loading raw train & test periods data ...')
    periods_test = pd.read_csv(config.PERIODS_TRAIN_DATA, parse_dates=["activation_date", "date_from", "date_to"],
                               nrows=config.RAW_DATA_ROWS)
    periods_train = pd.read_csv(config.PERIODS_TEST_DATA, parse_dates=["activation_date", "date_from", "date_to"],
                                nrows=config.RAW_DATA_ROWS)
    logger.info('Loading raw train & test periods data took: %s minutes' % round((time() - t0) / 60, 1))

    return periods_train, periods_test
