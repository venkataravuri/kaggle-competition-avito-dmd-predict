import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def main():
    FNAME = "feature_date"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    train, test = dl.load_data()

    logger.info("Generating activation date features ...")
    t0 = time()
    # Generating activation date features
    # Train data
    train['month'] = train['activation_date'].dt.month
    train['weekday'] = train['activation_date'].dt.weekday
    train['month_day'] = train['activation_date'].dt.day
    train['year_day'] = train['activation_date'].dt.dayofyear
    # Test data
    test['month'] = test['activation_date'].dt.month
    test['weekday'] = test['activation_date'].dt.weekday
    test['month_day'] = test['activation_date'].dt.day
    test['year_day'] = test['activation_date'].dt.dayofyear
    gc.collect()
    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, train[config.GENERATED_DATE_FEATURES])
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, test[config.GENERATED_DATE_FEATURES])
    gc.collect()


if __name__ == "__main__":
    main()
