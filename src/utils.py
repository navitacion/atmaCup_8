import os
import gc
import re
import sys
import random
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
from sklearn.model_selection import KFold


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    with np.errstate(invalid='ignore'):
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    # elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    #     df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def load_data(data_dir, down_sample=1.0, seed=0):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    train['id'] = np.arange(len(train))
    test['id'] = np.arange(len(test))
    test['Global_Sales'] = 0

    train['is_train'] = 1
    test['is_train'] = 0

    if down_sample < 1:
        train = train.sample(frac=down_sample, random_state=seed).reset_index(drop=True)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    return df


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p

