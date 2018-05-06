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
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def generate_figure_importance(model, logger):
    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    FNAME = config.FIGURES_DIR + 'feature_importance_xgb.png'
    logger.info('Saving feature importance at, ' + FNAME)
    plt.gcf().savefig(FNAME)


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
    train_raw = dl.load_train_data()
    # Load generated features
    train_features = load_combined_features(logger)

    train_column_names = list(train_features.columns.values)
    logger.info("Training set column names: " + str(train_column_names))

    # train_features = pd.concat([train_features, train_raw[config.NUMBER_FEATURES]], axis=1)
    logger.info('Final training data shape: %s' % str(train_features.shape))

    x_train, x_valid, y_train, y_valid = train_test_split(train_features, train_raw[config.TARGET_FEATURE],
                                                          test_size=0.20,
                                                          random_state=42)
    del train_raw
    del train_features
    gc.collect()
    lgtrain = lgb.Dataset(x_train, label=y_train, feature_name=train_column_names,
                          categorical_feature=config.ENCODED_CATEGORY_FEATURES)
    lgvalid = lgb.Dataset(x_valid, label=y_valid, feature_name=train_column_names,
                          categorical_feature=config.ENCODED_CATEGORY_FEATURES)

    t0 = time()
    lightgbm_model = lgb.train(config.LGBM_PARAMS, lgtrain, config.LGBM_NUM_ROUNDS,
                               valid_sets=lgvalid, verbose_eval=50,
                               early_stopping_rounds=config.LGBM_EARLY_STOPPING_ROUNDS)
    logger.info('Training LightGBM model took: %s minutes' % round((time() - t0) / 60, 1))

    # Save model
    t0 = time()
    MODEL_FILE_NAME = "lightgbm_model"
    model_file = os.path.join(config.DATA_MODELS_DIR, MODEL_FILE_NAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % model_file)
    lightgbm_model.save_model(model_file, num_iteration=lightgbm_model.best_iteration)
    logger.info('Saving %s lightgbm model took: %s minutes' % (MODEL_FILE_NAME, round((time() - t0) / 60, 1)))

    generate_figure_importance(lightgbm_model, logger)


if __name__ == "__main__":
    main()
