from utils import os_utils
import numpy as np

# ------------------------ PATH ------------------------
ROOT_DIR = ".."

DATA_DIR = "%s/data" % ROOT_DIR

DATA_RAW_DIR = "%s/raw" % DATA_DIR
DATA_FEATURES_DIR = "%s/features" % DATA_DIR
DATA_MODELS_DIR = "%s/models" % DATA_DIR
DATA_SUBMISSION_DIR = "%s/submissions" % DATA_DIR

LOG_DIR = "%s/logs" % ROOT_DIR
FIGURES_DIR = "%s/figures" % ROOT_DIR

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = "%s/train.csv" % DATA_RAW_DIR
TEST_DATA = "%s/test.csv" % DATA_RAW_DIR
PERIODS_TRAIN_DATA = "%s/periods_train.csv" % DATA_RAW_DIR
PERIODS_TEST_DATA = "%s/periods_test.csv" % DATA_RAW_DIR
SAMPLE_SUBMISSION_DATA = "%s/sample_submission.csv" % DATA_RAW_DIR

# ------------------------ PARAM ------------------------
RAW_DATA_ROWS = None  # MUST be None
MAX_TEXT_FEATURES = 5000
SVD_N_COMP = 3

LGBM_NUM_ROUNDS = 4000
LGBM_EARLY_STOPPING_ROUNDS = 100
LGBM_PARAMS = {'learning_rate': 0.03,
               'max_depth': 10,
               'boosting': 'gbdt',
               'objective': 'regression',
               'metric': ['rmse'],
               'is_training_metric': True,
               'seed': 19,
               'num_leaves': 128,
               'feature_fraction': 0.7,
               'bagging_fraction': 0.7,
               'bagging_freq': 5}

# ------------------------ FEATURES -----------------------
CATEGORY_FEATURES = ['user_id', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
                     'param_3', 'user_type', 'image_top_1', 'item_seq_number']
TEXT_FEATURES = ['title', 'description']
NUMBER_FEATURES = ['price']
ID_FEATURES = ['item_id', 'user_id', 'image']
DATE_FEATURES = ['activation_date']
GENERATED_DATE_FEATURES = ['month', 'weekday', 'month_day', 'year_day']
TARGET_FEATURE = 'deal_probability'

ENCODED_CATEGORY_FEATURES = ['enc_user_id', 'enc_region', 'enc_city', 'enc_parent_category_name', 'enc_category_name',
                             'enc_param_1', 'enc_param_2', 'enc_param_3', 'enc_user_type',
                             'enc_image_top_1', 'enc_item_seq_number']

AGGREGATE_COLUMNS = ['region', 'city', 'parent_category_name', 'category_name',
                     'image_top_1', 'user_type', 'item_seq_number', 'month_day', 'weekday']

AGGREGATE_DEAL_FEATURES = []
AGGREGATE_PRICE_FEATURES = []
for column in AGGREGATE_COLUMNS:
    AGGREGATE_DEAL_FEATURES.append(column + '_deal_probability_avg')
    AGGREGATE_DEAL_FEATURES.append(column + '_deal_probability_std')
    AGGREGATE_PRICE_FEATURES.append(column + '_price_avg')

ALL_FEATURES = CATEGORY_FEATURES + TEXT_FEATURES + NUMBER_FEATURES + ID_FEATURES + DATE_FEATURES

FEAT_FILE_SUFFIX = ".pkl"

# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [DATA_FEATURES_DIR]
DIRS += [DATA_SUBMISSION_DIR]
DIRS += [LOG_DIR]
DIRS += [DATA_MODELS_DIR]
DIRS += [FIGURES_DIR]

os_utils._create_dirs(DIRS)

# -------------- DTYPES ---------------

DTYPES = {
    'item_id': str,
    'user_id': str,
    'region': str,
    'city': str,
    'parent_category_name': str,
    'category_name': str,
    'param_1': str,
    'param_2': str,
    'param_3': str,
    'title': str,
    'description': str,
    'price': np.float32,
    'item_seq_number': np.uint32,
    'activation_date': object,  # in fact yyyy-mm-dd date
    'user_type': str,
    'image': str,
    'image_top_1': np.float32,
    'deal_probability': np.float32
}
