import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABCMeta, abstractmethod


# Basis -----------------------------------------------------------------------------------------------
class BaseModel(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError


# LightGBM -----------------------------------------------------------------------------------------------
class LGBMModel(BaseModel):
    def __init__(self, params):
        super(LGBMModel, self).__init__(params)

    def train(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(self.params,
                          train_data,
                          valid_sets=[valid_data, train_data],
                          valid_names=['eval', 'train'],
                          verbose_eval=1000,
                          )

        oof = model.predict(X_val, num_iteration=model.best_iteration)

        return oof, model


    def predict(self, X_test):
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        return pred


# CatBoost -----------------------------------------------------------------------------------------------
class CatBoostModel(BaseModel):
    def __init__(self, params):
        super(CatBoostModel, self).__init__(params)

    def train(self, X_train, y_train, X_val, y_val):
        train_data = Pool(X_train, label=y_train)
        valid_data = Pool(X_val, label=y_val)
        model = CatBoostRegressor(**self.params)
        model.fit(train_data,
                  eval_set=valid_data,
                  use_best_model=True)

        oof = model.predict(X_val)

        return oof, model

    def predict(self, X_test):
        pred = model.predict(X_test)
        return pred
