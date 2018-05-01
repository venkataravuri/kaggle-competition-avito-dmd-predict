from utils import os_utils

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
SAMPLE_SUBMISSION_DATA = "%s/sample_submission.csv" % DATA_RAW_DIR

# ------------------------ PARAM ------------------------
RAW_DATA_ROWS = 1000  # MUST be None
MAX_TEXT_FEATURES = 2000

LGBM_NUM_ROUNDS = 2000
LGBM_PARAMS = {'learning_rate': 0.05,
          'max_depth': 7,
          'boosting': 'gbdt',
          'objective': 'regression',
          'metric': ['auc','rmse'],
          'is_training_metric': True,
          'seed': 19,
          'num_leaves': 128,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.8,
          'bagging_freq': 5}

# ------------------------ FEATURES -----------------------
CATEGORY_FEATURES = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
                     'param_3', 'user_type', 'image_top_1']
TEXT_FEATURES = ['title', 'description']
NUMBER_FEATURES = ['price', 'item_seq_number']
ID_FEATURES = ['item_id', 'user_id', 'image']
DATE_FEATURES = ['activation_date']
GENERATED_DATE_FEATURES = ['month', 'weekday', 'month_day', 'year_day']
TARGET_FEATURE = 'deal_probability'

ENCODED_CATEGORY_FEATURES = ['enc_region', 'enc_city', 'enc_parent_category_name', 'enc_category_name', 'enc_param_1',
                             'enc_param_2',
                             'enc_param_3', 'enc_user_type', 'enc_image_top_1']

ALL_FEATURES = CATEGORY_FEATURES + TEXT_FEATURES + NUMBER_FEATURES + ID_FEATURES + DATE_FEATURES

FEAT_FILE_SUFFIX = ".pkl"

# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [DATA_FEATURES_DIR]
DIRS += [DATA_SUBMISSION_DIR]
DIRS += [LOG_DIR]
DIRS += [DATA_MODELS_DIR]

os_utils._create_dirs(DIRS)
