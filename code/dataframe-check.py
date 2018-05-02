# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import config

# df = pd.read_csv(constants.TRAIN_RAW_FILE, nrows=None,
#                     usecols=constants.TRAIN_STANDARD_COLUMNS,
#                     dtype=constants.DTYPES)

df = pd.read_csv(config.DATA_SUBMISSION_DIR + "/submission_lightgbm.csv")
print(df.shape)
print(df.head(5))
print(df.isna().any())
# df['datetime'] = pd.to_datetime(df['click_time'])
# df['dow'] = df['datetime'].dt.dayofweek
# df["doy"] = df["datetime"].dt.dayofyear
# print("dow: " + str(df['dow'].value_counts()))
# print("doy: " + str(df['doy'].value_counts()))
print(df['deal_probability'].value_counts())
print(df.info())
#print(df.loc[18789379])

