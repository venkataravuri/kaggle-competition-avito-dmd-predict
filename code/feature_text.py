import gc
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import config
from utils import logging_utils, pkl_utils, time_utils
import data_loader as dl
from time import time

nltk.download('stopwords')
from nltk.corpus import stopwords

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# -------------------------- Main --------------------------
now = time_utils._timestamp()


def main():
    FNAME = "feature_text"
    logname = "%s_%s.log" % (FNAME, now)
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    stop_words = set(stopwords.words('russian'))

    train, test = dl.load_data()

    logger.info("Generating activation date features ...")
    t0 = time()
    # Generating text features for title
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=config.MAX_TEXT_FEATURES)
    vectorizer.fit(train['title'])
    train_title_tfidf = vectorizer.transform(train['title'])
    test_title_tfidf = vectorizer.transform(test['title'])
    logger.info(FNAME + ' took: %s minutes' % round((time() - t0) / 60, 1))

    logger.info('Total vocabulary in title: %s' % len(vectorizer.vocabulary_))
    logger.info('Total stop words in title: %s' % len(vectorizer.stop_words_))
    logger.info('Total text features generated from title: %s' % len(vectorizer.get_feature_names()))
    logger.info('Text features list(10) in title: %s' % vectorizer.get_feature_names()[:10])
    gc.collect()

    # save data
    train_fname = os.path.join(config.DATA_FEATURES_DIR, "train_" + FNAME + config.FEAT_FILE_SUFFIX)
    test_fname = os.path.join(config.DATA_FEATURES_DIR, "test_" + FNAME + config.FEAT_FILE_SUFFIX)
    logger.info("Save to %s" % train_fname)
    pkl_utils._save(train_fname, train_title_tfidf)
    logger.info("Save to %s" % test_fname)
    pkl_utils._save(test_fname, test_title_tfidf)
    gc.collect()


if __name__ == "__main__":
    main()
