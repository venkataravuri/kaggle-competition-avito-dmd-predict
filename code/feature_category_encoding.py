import gc
import os
import pandas as pd
from category_encoders.hashing import HashingEncoder
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time
from sklearn import preprocessing

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def main():
    logname = "feature_category_encoding_%s.log" % now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    FNAME = "feature_category_encoding"

    train, test = dl.load_data()

    train_length = len(train.index)

    train["image_top_1"].fillna(-999, inplace=True)
    test["image_top_1"].fillna(-999, inplace=True)

    logger.info("Combine Train and Test")
    df = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()
    logger.info('All Data shape: {} Rows, {} Columns'.format(*df.shape))

    logger.info("Encoding category features ...")
    t0 = time()
    # Encode category features using Hashing Encoder

    logger.info("category features:  %s" % len(config.CATEGORY_FEATURES))
    logger.info("encoded category features:  %s" % len(config.ENCODED_CATEGORY_FEATURES))

    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for feature in config.CATEGORY_FEATURES:
        df["enc_" + feature] = lbl.fit_transform(df[feature].astype(str))

    # Dropping category features.
    # train.drop(config.CATEGORY_FEATURES, axis=1, inplace=True)
    # test.drop(config.CATEGORY_FEATURES, axis=1, inplace=True)
    gc.collect()
    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, df[config.ENCODED_CATEGORY_FEATURES][train_length:])
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, df[config.ENCODED_CATEGORY_FEATURES][:train_length])


if __name__ == "__main__":
    main()
