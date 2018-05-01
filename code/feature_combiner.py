import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
from time import time
from scipy.sparse import hstack

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# -------------------------- Main --------------------------
now = time_utils._timestamp()


class Combiner:
    def __init__(self, feature_names, logger):
        self.feature_names = feature_names
        self.logger = logger
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()

    def load_feature(self, feature_name):
        t0 = time()
        train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + feature_name + config.FEAT_FILE_SUFFIX)
        test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + feature_name + config.FEAT_FILE_SUFFIX)
        self.logger.info('Loading %s features' % feature_name)
        train = pkl_utils._load(train_fname)
        test = pkl_utils._load(test_fname)
        self.logger.info('Loading %s features took: %s minutes' % (feature_name, round((time() - t0) / 60, 1)))

        return train, test

    def combine(self):
        for feature_name in self.feature_names:
            train, test = self.load_feature(feature_name)
            self.x_train, self.x_test = hstack([self.x_train, train]), hstack([self.x_test, test])
            self.logger.info('Combined train shape - %s, test shape - %s' % (self.x_train.shape, self.x_test.shape))

        return self

    def save(self):
        t0 = time()
        FNAME = "combine_features"
        train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
        test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
        self.logger.info("Save to %s" % train_fname)
        pkl_utils._save(train_fname, self.x_train)
        self.logger.info("Save to %s" % test_fname)
        pkl_utils._save(test_fname, self.x_test)
        self.logger.info('Saving %s features took: %s minutes' % (FNAME, round((time() - t0) / 60, 1)))


def main():
    FNAME = "feature_combiner"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    feature_names = ['feature_category_encoding',
                     'feature_date',
                     'feature_text']

    combiner = Combiner(feature_names=feature_names,
                        logger=logger)

    logger.info("Combining features ...")
    combiner.combine()
    combiner.save()


if __name__ == "__main__":
    main()
