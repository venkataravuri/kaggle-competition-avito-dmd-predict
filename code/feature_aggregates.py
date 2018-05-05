import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time
from tqdm import tqdm

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def agg_deal_probability_features(train, test, aggregate_columns):
    for column in aggregate_columns:
        gp = train.groupby(column)[config.TARGET_FEATURE]
        mean = gp.mean()
        std = gp.std()
        train[column + '_deal_probability_avg'] = train[column].map(mean)
        train[column + '_deal_probability_std'] = train[column].map(std)
        test[column + '_deal_probability_avg'] = test[column].map(mean)
        test[column + '_deal_probability_std'] = test[column].map(std)


def agg_price_features(train, test, aggregate_columns):
    for column in aggregate_columns:
        gp = train.groupby(column)['price']
        mean = gp.mean()
        train[column + '_price_avg'] = train[column].map(mean)
        test[column + '_price_avg'] = test[column].map(mean)


def main():
    FNAME = "feature_aggregates"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    train, test = dl.load_data()

    train['weekday'] = train['activation_date'].dt.weekday
    train['month_day'] = train['activation_date'].dt.day
    test['weekday'] = test['activation_date'].dt.weekday
    test['month_day'] = test['activation_date'].dt.day

    logger.info("Train shape: %s & Test shape: %s" % (train.shape, test.shape))
    logger.info("Generating aggregate features ...")
    t0 = time()
    # Generating aggregate features
    agg_deal_probability_features(train, test, config.AGGREGATE_COLUMNS)
    agg_price_features(train, test, config.AGGREGATE_COLUMNS)
    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))
    logger.info("Train shape: %s & Test shape: %s" % (train.shape, test.shape))
    gc.collect()

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, train[config.AGGREGATE_DEAL_FEATURES + config.AGGREGATE_PRICE_FEATURES])
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, test[config.AGGREGATE_DEAL_FEATURES + config.AGGREGATE_PRICE_FEATURES])
    gc.collect()


if __name__ == "__main__":
    main()
