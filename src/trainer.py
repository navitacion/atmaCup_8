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


class Trainer:
    def __init__(self, model, id_col, tar_col, features, cv, criterion, experiment):
        self.model = model
        self.id_col = id_col
        self.tar_col = tar_col
        self.features = features
        self.cv = cv
        self.criterion = criterion
        self.experiment = experiment
        # Log params
        self.experiment.log_parameters(self.model.params)


    def _prepare_data(self, df, mode='fit'):
        """
        データの分割、準備
        self.X_train, self.y_train, self.train_id, self.X_test, self.test_id
        """
        assert 'is_train' in df.columns, 'Not contained is_train'
        # train, testに分割
        train = df[df['is_train'] == 1].reset_index(drop=True)
        test = df[df['is_train'] == 0].reset_index(drop=True)

        if self.features is None:
            self.features = [c for c in df.columns if c not in ['is_train', self.id_col, self.tar_col]]
        else:
            self.features = [c for c in self.features if c not in ['is_train', self.id_col, self.tar_col]]

        if mode == 'fit':
            # Train
            self.X_train = train[self.features].values
            self.y_train = train[self.tar_col].values
            self.y_train = self._transform_value(self.y_train, mode='forward')
            self.train_id = train[self.id_col].values
            self.publisher = train['Publisher']

        if mode == 'predict':
            # Test
            self.X_test = test[self.features].values
            self.test_id = test[self.id_col].values


    def _transform_value(self, v, mode='forward'):
        """
        目的変数の値を変える変数
        forwardは変更する、backwardはもとに戻す関数
        """
        if mode == 'forward':
            out = np.log1p(v)
        elif mode == 'backward':
            v = np.where(v < 0, 0, v)
            out = np.expm1(v)
        else:
            out = v

        return out


    def _train_cv(self):
        """
        cvごとに学習を行う
        """
        self.models = []
        self.oof_pred = np.zeros(len(self.y_train))
        self.oof_y = np.zeros(len(self.y_train))

        # Publisherを用いたGroupKFold
        # Reference https://www.guruguru.science/competitions/13/discussions/cc7167cb-3627-448a-b9eb-7afcd29fd122/
        for i, (_trn_idx, _val_idx) in enumerate(self.cv.split(self.publisher.unique())):
            tr_groups, val_groups = self.publisher.unique()[_trn_idx], self.publisher.unique()[_val_idx]
            trn_idx = self.publisher.isin(tr_groups)
            val_idx = self.publisher.isin(val_groups)

            X_trn, y_trn = self.X_train[trn_idx], self.y_train[trn_idx]
            X_val, y_val = self.X_train[val_idx], self.y_train[val_idx]

            oof, model = self.model.train(X_trn, y_trn, X_val, y_val)

            # Score
            oof = self._transform_value(oof, mode='backward')
            y_val = self._transform_value(y_val, mode='backward')
            score = self.criterion(y_val, oof)

            # Logging
            self.experiment.log_metric('Fold_score', score, step=i + 1)
            print(f'Fold {i + 1}  Score: {score:.3f}')
            self.oof_pred[val_idx] = oof
            self.oof_y[val_idx] = y_val
            self.models.append(model)


    def _train_end(self):
        """
        学習の最後に呼び出し
        oofの作成と記録
        :return:
        """
        # Log params
        self.oof_score = self.criterion(self.oof_y, self.oof_pred)
        self.experiment.log_metric('Score', self.oof_score)
        print(f'All Score: {self.oof_score:.3f}')

        oof = pd.DataFrame({
            self.id_col: self.train_id,
            self.tar_col: self._transform_value(self.oof_pred, mode='backward')
        })

        oof = oof.sort_values(by='id')

        # 0以下のものは0とする
        oof.loc[oof[self.tar_col] < 0, self.tar_col] = 0

        # Comet_MLに保存
        sub_name = f'oof_score_{self.oof_score:.4f}.csv'
        oof[[self.tar_col]].to_csv(os.path.join(sub_name), index=False)
        self.experiment.log_asset(file_data=sub_name, file_name=sub_name)
        os.remove(sub_name)


    def _predict_cv(self):
        """
        cvごとの推論を行う
        """
        assert len(self.models), 'You Must Train Something Model'
        self.preds = np.zeros(len(self.test_id))

        for m in self.models:
            pred = m.predict(self.X_test)
            self.preds += pred

        self.preds /= len(self.models)
        self.preds = self._transform_value(self.preds, mode='backward')


    def _predict_end(self):
        """
        推論の最後に呼び出し
        提出用のファイルを作成、保存
        :return:
        """
        sub = pd.DataFrame({
            self.id_col: self.test_id,
            self.tar_col: self.preds
        })

        sub = sub.sort_values(by='id')

        # 0以下のものは0とする
        sub.loc[sub[self.tar_col] < 0, self.tar_col] = 0

        # CometMLに保存
        sub_name = f'sub_score_{self.oof_score:.4f}.csv'
        sub[[self.tar_col]].to_csv(os.path.join(sub_name), index=False)
        self.experiment.log_asset(file_data=sub_name, file_name=sub_name)
        os.remove(sub_name)

    def fit(self, df):
        self._prepare_data(df, mode='fit')
        self._train_cv()
        self._train_end()


    def predict(self, df):
        self._prepare_data(df, mode='predict')
        self._predict_cv()
        self._predict_end()


    def get_feature_importance(self):
        assert len(self.models) != 0, "You Must Train Model!!"
        feat_imp = np.zeros(len(self.features))

        for m in self.models:
            feat_imp += m.feature_importance()

        feat_imp = feat_imp / len(self.models)

        feat_imp_df = pd.DataFrame({
            'feature': self.features,
            'importance': feat_imp
        })
        feat_imp_df.sort_values(by='importance', ascending=False, inplace=True)
        feat_imp_df.reset_index(drop=True, inplace=True)
        feat_imp_df.to_csv('feature_importance.csv', index=False)
        self.experiment.log_asset(file_data='feature_importance.csv', file_name='feature_importance.csv')
        os.remove('feature_importance.csv')
