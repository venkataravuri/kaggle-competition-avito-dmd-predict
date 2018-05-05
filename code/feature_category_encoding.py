import gc
import os
import pandas as pd
from category_encoders.hashing import HashingEncoder
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time


# -------------------------- Main --------------------------
now = time_utils._timestamp()


def main():
    logname = "feature_category_encoding_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    FNAME = "feature_category_encoding"

    train, test = dl.load_data()

    logger.info("Encoding category features ...")
    t0 = time()
    # Encode category features using Hashing Encoder
    he = HashingEncoder()
    he.fit(train[config.CATEGORY_FEATURES].values)
    train[config.ENCODED_CATEGORY_FEATURES] = he.transform(train[config.CATEGORY_FEATURES].values)
    test[config.ENCODED_CATEGORY_FEATURES] = he.transform(test[config.CATEGORY_FEATURES].values)
    # Dropping cateogory features.
    train.drop(config.CATEGORY_FEATURES, axis=1, inplace=True)
    test.drop(config.CATEGORY_FEATURES, axis=1, inplace=True)
    gc.collect()
    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, train[config.ENCODED_CATEGORY_FEATURES])
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, test[config.ENCODED_CATEGORY_FEATURES])


if __name__ == "__main__":
    main()
