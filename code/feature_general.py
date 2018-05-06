import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time
import string
import numpy as np

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def generate_count_features(name, feature_df, df):
    # Fill Not a Number with blank
    df[name].fillna('', inplace=True)
    df[name] = df[name].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
    feature_df[name + '_length'] = df[name].apply(lambda x: len(str(x)))
    feature_df[name + '_char_count'] = df[name].apply(len)
    feature_df[name + '_word_count'] = df[name].apply(lambda x: len(x.split()))
    feature_df[name + '_word_density'] = feature_df[name + '_char_count'] / (feature_df[name + '_word_count'] + 1)
    feature_df[name + '_punctuation_count'] = df[name].apply(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))


def main():
    FNAME = "feature_general"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    logger.info("Generating time period feature ...")
    train, test = dl.load_data()
    periods_train, periods_test = dl.load_periods_data()

    t0 = time()
    # Generating general features
    train_general = pd.DataFrame()
    test_general = pd.DataFrame()

    logger.info("Generating general features ...")

    train = train.merge(periods_train[['item_id', 'date_to', 'date_from']], how='left', on=['item_id'])
    test = test.merge(periods_test[['item_id', 'date_to', 'date_from']], how='left', on=['item_id'])
    logger.info('Train  shape: %s & Test shape: %s' % (train.shape, test.shape))

    # https: // stackoverflow.com / questions / 37840812 / pandas - subtracting - two - date - columns - and -the - result - being - an - integer
    train_general['total_period'] = train['date_to'].sub(train['date_from'], axis=0)
    train_general['total_period'] = train_general['total_period'] / np.timedelta64(1, 'D')
    train_general['total_period'].fillna(0, inplace=True)
    test_general['total_period'] = test['date_to'].sub(test['date_from'], axis=0)
    test_general['total_period'] = test_general['total_period'] / np.timedelta64(1, 'D')
    test_general['total_period'].fillna(0, inplace=True)

    generate_count_features('title', train_general, train)
    generate_count_features('title', test_general, test)

    generate_count_features('description', train_general, train)
    generate_count_features('description', test_general, test)

    train_general['log_price'] = np.log(train["price"] + 0.001)
    train_general['log_price'].fillna(-999, inplace=True)

    test_general['log_price'] = np.log(test["price"] + 0.001)
    test_general['log_price'].fillna(-999, inplace=True)

    train['has_image'] = train['image'].isnull().astype(int)
    test['has_image'] = test['image'].isnull().astype(int)

    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))
    del train
    del test
    gc.collect()

    logger.info('Train general shape: %s & Test general shape: %s' % (train_general.shape, test_general.shape))

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, train_general)
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, test_general)
    gc.collect()


if __name__ == "__main__":
    main()
