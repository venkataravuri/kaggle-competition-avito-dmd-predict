import gc
import os
import pandas as pd
import config
from utils import logging_utils, pkl_utils, time_utils
from time import time
from scipy.sparse import csr_matrix, hstack
import data_loader as dl
import lightgbm as lgb
import zipfile

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def load_combined_features(logger):
    t0 = time()
    FNAME = "combine_features"
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Loading %s ..." % test_fname)
    test = pkl_utils._load(test_fname)
    logger.info('Loading %s features took: %s minutes' % (FNAME, round((time() - t0) / 60, 1)))

    return test


def load_model(logger):
    t0 = time()
    MODEL_FILE_NAME = "lightgbm_model"
    model_file = os.path.join(config.DATA_MODELS_DIR, MODEL_FILE_NAME + config.FEAT_FILE_SUFFIX)
    logger.info("Loading %s ..." % model_file)
    lightgbm_model = lgb.Booster(model_file=model_file)  # init model
    logger.info('Loading %s model took: %s minutes' % (MODEL_FILE_NAME, round((time() - t0) / 60, 1)))

    return lightgbm_model


def main():
    FNAME = "model_predict_lgbm"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    # Load raw data
    test_raw = dl.load_test_data()
    gc.collect()
    # Load generated features
    test_features = load_combined_features(logger)

    #test_features = pd.concat([test_features, test_raw[config.NUMBER_FEATURES]], axis=1)
    logger.info('Final test data shape: %s' % str(test_features.shape))

    lightgbm_model = load_model(logger)

    t0 = time()
    pred = lightgbm_model.predict(test_features)

    submission = pd.read_csv(config.SAMPLE_SUBMISSION_DATA, nrows=config.RAW_DATA_ROWS)
    submission['deal_probability'] = pred
    submission['deal_probability'].clip(0.0, 1.0, inplace=True)
    submission_file = os.path.join(config.DATA_SUBMISSION_DIR, "submission_lightgbm.csv")
    submission.to_csv(submission_file, index=False)

    # Compress (zip) submission file.
    submission_zip_file = os.path.join(config.DATA_SUBMISSION_DIR, "submission_lightgbm.csv.zip")
    submission_zip = zipfile.ZipFile(submission_zip_file, 'w')
    submission_zip.write(submission_file, arcname="submission_lightgbm.csv", compress_type=zipfile.ZIP_DEFLATED)
    submission_zip.close()
    logger.info('LightGBM submission file generation took: %s minutes' % round((time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
