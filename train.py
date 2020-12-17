import os
import gc
import datetime
import itertools
import numpy as np
from comet_ml import Experiment
import hydra
from omegaconf import DictConfig

import pandas as pd
import cudf

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error
import category_encoders as ce

from src.utils import load_data, seed_everything, reduce_mem_usage, to_pickle, unpickle
from src.preprocessing import prep_text, tfidf_vectorizer, idx_col_count_encode, target_encode

from src.trainer import Trainer
from src.models import LGBMModel, CatBoostModel

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)


def RMLSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    score = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return score


def preprocessing(df, cfg):
    # 変なデータは削除
    df = df[df['Name'] != 'Strongest Tokyo University Shogi DS']
    df = df.reset_index(drop=True)
    df.loc[df['Name'] == 'Imagine: Makeup Artist', 'Year_of_Release'] = np.nan

    # Nameを前処理
    df['Name'] = df['Name'].fillna('unknown')
    df = prep_text(df)

    # TODO 発売から経過年数
    df['fe_diff_from_now'] = 2020 - df['Year_of_Release']

    # TODO tbdかどうか
    df['fe_is_tbd'] = df['User_Score'].apply(lambda x: 1 if x == 'tbd' else 0)

    # TODO ハードが携帯端末かどうか
    mobile = ['DS', 'PSP', 'GBA', '3DS', 'PSV', 'GB']
    df['fe_is_mobile'] = df['Platform'].apply(lambda x: 1 if x in mobile else 0)

    # TODO 年代をカテゴリ化
    dev_year = [0, 1992, 1999, 2004, 2009, 2021]
    df['fe_cat_year'] = pd.cut(df['Year_of_Release'], dev_year, labels=False)
    df.loc[(df['Year_of_Release'].isnull()), 'fe_cat_year'] = np.nan

    # TODO ユーザースコアのtbdを-1とし、数値に変換する
    df['User_Score'] = df['User_Score'].replace('tbd', -1)
    df['User_Score'] = df['User_Score'].astype(float)

    # TODO ターゲット変数の外れ値を除外する
    # df = df[df['Global_Sales'] < 1000]

    # TODO スコアと数の掛け合わせた変数
    df['fe_product_critic_score_count'] = df['Critic_Score'] * df['Critic_Count']
    df['fe_product_user_score_count'] = df['User_Score'] * df['User_Count']

    # TODO ソフト名をテキストマイニング
    print('Text Vectorizing')
    df = tfidf_vectorizer(df, tar_col='Name',
                          max_features=cfg.data.vec_max_features,
                          ngram_range=(1, 1),
                          n_components=cfg.data.vec_n_components,
                          type='svd')

    df = tfidf_vectorizer(df, tar_col='Name',
                          max_features=cfg.data.vec_max_features,
                          ngram_range=(2, 2),
                          n_components=cfg.data.vec_n_components,
                          type='svd')

    df = tfidf_vectorizer(df, tar_col='Name',
                          max_features=cfg.data.vec_max_features,
                          ngram_range=(3, 3),
                          n_components=cfg.data.vec_n_components,
                          type='svd')

    df = tfidf_vectorizer(df, tar_col='Name',
                          max_features=cfg.data.vec_max_features,
                          ngram_range=(1, 3),
                          n_components=cfg.data.vec_n_components,
                          type='svd')

    # TODO テキストマイニング結果をKMeans
    tar_cols = [c for c in df.columns if c.startswith('fe_tfidf')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_kmeans'] = kmeans.fit_predict(df[tar_cols])

    # TODO Publisher単位の分布個数に応じた変数
    print('Publisher')
    all_var_unique = 0
    tar_cols = ['Year_of_Release', 'Platform', 'Genre', 'Developer', 'Rating']
    for c in tar_cols:
        all_var_unique += df[c].nunique()
    df = idx_col_count_encode(df, tar_cols, idx_col='Publisher', n_components=int(all_var_unique * 0.2), type='svd')

    # TODO Developer単位の分布個数に応じた変数
    print('Developer')
    all_var_unique = 0
    tar_cols = ['Year_of_Release', 'Platform', 'Genre', 'Publisher', 'Rating']
    for c in tar_cols:
        all_var_unique += df[c].nunique()
    df = idx_col_count_encode(df, tar_cols, idx_col='Developer', n_components=int(all_var_unique * 0.2), type='svd')

    # TODO Publisher単位の分布個数結果をKMeans
    print('KMeans1')
    tar_cols = [c for c in df.columns if c.startswith('fe_Publisher_count')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_Publisher_count_kmeans'] = kmeans.fit_predict(df[tar_cols].fillna(0))

    # TODO Developer単位の分布個数結果をKMeans
    print('KMeans2')
    tar_cols = [c for c in df.columns if c.startswith('fe_Developer_count')]
    kmeans = KMeans(n_clusters=10)
    df['fe_cat_Developer_count_kmeans'] = kmeans.fit_predict(df[tar_cols].fillna(0))



    # TODO ある単語が入っているかどうかの変数
    target_name = ['soccer', 'dragon', 'wars', 'ds', 'battle', 'disney', 'lego',
                   'collection', 'party', 'ultimate', 'edition', 'baseball', 'fantasy',
                   'gundam', 'legend', 'mario', 'ninja', 'monster', 'sonic', 'samurai',
                   'tennis', 'batman', 'harry', 'yugioh', 'assassin']
    for t in target_name:
        df[f'fe_is_{t}_in_name'] = df['Name'].apply(lambda x: 1 if t in x.lower() else 0)

    # TODO Groupbyでいろいろ集約

    # Cudf
    df = cudf.from_pandas(df)

    group_cols = ['Platform', 'Genre', 'Developer', 'Rating', 'fe_cat_year', 'fe_cat_kmeans',
                  'fe_cat_Publisher_count_kmeans', 'fe_cat_Developer_count_kmeans']
    value_cols = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
                  'fe_product_critic_score_count', 'fe_product_user_score_count']
    aggs = ['mean', 'sum', 'std', 'max', 'min', 'nunique']
    # Categoryにしておく
    for g in group_cols:
        df[g] = df[g].astype('category')

    print('Groupby1')
    for g, v, agg in itertools.product(group_cols, value_cols, aggs):
        if v == 'User_Score':
            # User_Scoreがtbdの場合は除く
            temp = df[df[v] > 0]
        else:
            temp = df.copy()

        col_name = f'fe_group_{g}_{v}_{agg}'
        group = temp[[g, v]].groupby(g)[v].agg(agg).reset_index()
        group = group.rename(columns={v: col_name})
        df = df.merge(group, on=[g], how='left')

        # 発売年を含める
        # 微妙かも
        for y in [0, 1, 2, 3]:
            col_name = f'fe_group_{y}year_{g}_{v}_{agg}'
            group = df.groupby(['Year_of_Release', g])[v].agg(agg).reset_index()
            # yの値で足すことで、その年の前の値を表現する
            group['Year_of_Release'] += y
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=['Year_of_Release', g], how='left')

    # TODO groupの組み合わせを考慮してGroupby
    print('Groupby2')
    for v, agg in itertools.product(value_cols, aggs):
        for g in itertools.combinations(group_cols, 2):
            col_name = f'fe_group_{g[0]}_{g[1]}_{v}_{agg}'
            group = df[[g[0], g[1], v]].groupby([g[0], g[1]])[v].agg(agg).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=[g[0], g[1]], how='left')

    # TODO groupの組み合わせ*3を考慮してGroupby
    print('Groupby3')
    for v, agg in itertools.product(value_cols, aggs):
        for g in itertools.combinations(group_cols, 3):
            col_name = f'fe_group_{g[0]}_{g[1]}_{g[2]}_{v}_{agg}'
            group = df[[g[0], g[1], g[2], v]].groupby([g[0], g[1], g[2]])[v].agg(agg).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=[g[0], g[1], g[2]], how='left')

    # TODO groupの組み合わせ*4を考慮してGroupby
    print('Groupby4')
    for v, agg in itertools.product(value_cols, aggs):
        for g in itertools.combinations(group_cols, 4):
            col_name = f'fe_group_{g[0]}_{g[1]}_{g[2]}_{g[3]}_{v}_{agg}'
            group = df[[g[0], g[1], g[2], g[3], v]].groupby([g[0], g[1], g[2], g[3]])[v].agg(agg).reset_index()
            group = group.rename(columns={v: col_name})
            df = df.merge(group, on=[g[0], g[1], g[2], g[3]], how='left')

    # Cudf
    df = df.to_pandas()

    # TODO Frequency Encoding
    # trainとtestで分布が全く違うPublisherを入れるとスコア悪くなる！
    group_cols = ['Platform', 'Genre', 'Developer', 'Rating', 'Name']
    for g in group_cols:
        col_name = f'fe_freq_{g}'
        freq = df[df['is_train'] == 1][g].value_counts()
        df[col_name] = df[g].map(freq)


    # ---------- 下記はfeature_enginneringを一通りやったあとの処理 -------------------------------------------
    not_use_col = ['id', 'Global_Sales', 'is_train']

    # カテゴリ変数に
    cols = [c for c in df.select_dtypes(include=['object']).columns if c not in not_use_col]
    cols += [c for c in df.columns if c.startswith('fe_is_')]
    cols += [c for c in df.columns if c.startswith('fe_cat_')]
    # 重複を削除
    cols = list(set(cols))
    for c in cols:
        df[c] = df[c].astype('category')

    # カテゴリ変数はラベルエンコーダで変換
    object_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = ce.OrdinalEncoder(cols=object_columns, handle_unknown='impute')
    df = encoder.fit_transform(df)

    return df


@hydra.main('config.yml')
def main(cfg: DictConfig):
    print('atmaCup #8 Model Training')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    data_dir = './input'

    seed_everything(cfg.data.seed)

    experiment = Experiment(api_key=cfg.exp.api_key,
                            project_name=cfg.exp.project_name)

    experiment.log_parameters(dict(cfg.data))

    # Load Data  ####################################################################################
    if cfg.exp.use_pickle:
        # pickleから読み込み
        df = unpickle('./input/data_a.pkl')

    else:
        df = load_data(data_dir, down_sample=1.0, seed=cfg.data.seed)
        # Preprocessing
        print('Preprocessing')
        df = preprocessing(df, cfg)

        # pickle形式で保存
        to_pickle('./input/data.pkl', df)
        try:
            experiment.log_asset(file_data='./input/data.pkl', file_name='data.pkl')
        except:
            pass

    # Feature Extract  ###########################################################################
    features = pd.read_csv(os.path.join(data_dir, 'feature_importance_a.csv'))

    features = features.iloc[:3000]['feature'].tolist()

    # Config  ####################################################################################
    del_tar_col = [
        'Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Publisher'
    ]
    id_col = 'id'
    tar_col = 'Global_Sales'
    criterion = RMLSE
    cv = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)

    # Model  ####################################################################################
    model = None
    if cfg.exp.model == 'lgb':
        model = LGBMModel(dict(cfg.lgb))
    elif cfg.exp.model == 'cat':
        model = CatBoostModel(dict(cfg.cat))

    # Train & Predict  ##############################################################################
    trainer = Trainer(model, id_col, tar_col, features, cv, criterion, experiment)
    trainer.fit(df)
    trainer.predict(df)
    trainer.get_feature_importance()


if __name__ == '__main__':
    main()
