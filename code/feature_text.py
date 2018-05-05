import gc
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time

nltk.download('stopwords')
from nltk.corpus import stopwords


# -------------------------- Main --------------------------
now = time_utils._timestamp()


def main():
    FNAME = "feature_text"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    stop_words = set(stopwords.words('russian'))

    train, test = dl.load_data()

    logger.info("Generating title & description text features ...")
    t0 = time()
    # Generating text features for title
    tfidf_title = TfidfVectorizer(stop_words=stop_words, max_features=config.MAX_TEXT_FEATURES)
    tfidf_description = TfidfVectorizer(stop_words=stop_words, max_features=config.MAX_TEXT_FEATURES)

    train['description'] = train['description'].fillna(' ')
    test['description'] = test['description'].fillna(' ')
    train['title'] = train['title'].fillna(' ')
    test['title'] = test['title'].fillna(' ')
    tfidf_title.fit(pd.concat([train['description'], train['description']]))
    tfidf_description.fit(pd.concat([test['title'], test['title']]))

    train_title_tfidf = tfidf_title.transform(train['title'])
    test_title_tfidf = tfidf_title.transform(test['title'])

    train_description_tfidf = tfidf_description.transform(train['description'])
    test_description_tfidf = tfidf_description.transform(test['description'])

    svd_title = TruncatedSVD(n_components=config.SVD_N_COMP, algorithm='arpack')
    svd_title.fit(tfidf_title.transform(pd.concat([train['title'], test['title']])))

    svd_description = TruncatedSVD(n_components=config.SVD_N_COMP, algorithm='arpack')
    svd_description.fit(tfidf_description.transform(pd.concat([train['description'], test['description']])))

    train_description_svd = pd.DataFrame(svd_description.transform(train_description_tfidf))
    test_description_svd = pd.DataFrame(svd_description.transform(test_description_tfidf))
    train_description_svd.columns = ['svd_description_' + str(i + 1) for i in range(config.SVD_N_COMP)]
    test_description_svd.columns = ['svd_description_' + str(i + 1) for i in range(config.SVD_N_COMP)]

    train_title_svd = pd.DataFrame(svd_title.transform(train_title_tfidf))
    test_title_svd = pd.DataFrame(svd_title.transform(test_title_tfidf))
    train_title_svd.columns = ['svd_title_' + str(i + 1) for i in range(config.SVD_N_COMP)]
    test_title_svd.columns = ['svd_title_' + str(i + 1) for i in range(config.SVD_N_COMP)]
    gc.collect()

    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))

    logger.info('Train SVD title shape: %s & Test SVD title shape: %s' % (train_title_svd.shape, test_title_svd.shape))
    logger.info('Train SVD description shape: %s & Test SVD description shape: %s' % (
        train_description_svd.shape, test_description_svd.shape))

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname,  pd.concat([train_title_svd, train_description_svd], axis=1))
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, pd.concat([test_title_svd, test_description_svd], axis=1))
    gc.collect()


if __name__ == "__main__":
    main()
