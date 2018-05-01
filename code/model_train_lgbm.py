import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
from time import time
from scipy.sparse import csr_matrix, hstack
import data_loader as dl
import lightgbm as lgb
from sklearn.model_selection import train_test_split

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def load_combined_features(logger):
    t0 = time()
    FNAME = "combine_features"
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Loading %s ..." % train_fname)
    train = pkl_utils._load(train_fname)
    logger.info('Loading %s features took: %s minutes' % (FNAME, round((time() - t0) / 60, 1)))

    return train


def main():
    FNAME = "model_train_lgbm"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    # Load raw data
    train_raw, test_raw = dl.load_data()
    del test_raw
    gc.collect()
    # Load generated features
    train_features = load_combined_features(logger)

    x_train_csr = csr_matrix(hstack([train_raw[config.NUMBER_FEATURES], train_features]))

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_csr, train_raw[config.TARGET_FEATURE], test_size=0.20,
                                                          random_state=42)
    gc.collect()

    t0 = time()
    lightgbm_model = lgb.train(config.LGBM_PARAMS, lgb.Dataset(x_train, label=y_train), config.LGBM_NUM_ROUNDS,
                               valid_sets=lgb.Dataset(x_valid, label=y_valid), verbose_eval=50,
                               early_stopping_rounds=20)
    logger.info('Training LightGBM model took: %s minutes' % round((time() - t0) / 60, 1))

    # Save model
    t0 = time()
    MODEL_FILE_NAME = "lightgbm_model"
    model_file = os.path.join(config.DATA_MODELS_DIR, MODEL_FILE_NAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % model_file)
    pkl_utils._save(model_file, lightgbm_model)
    logger.info('Saving %s lightgbm model took: %s minutes' % (MODEL_FILE_NAME, round((time() - t0) / 60, 1)))


if __name__ == "__main__":
    main()
