import gc
import re
import numpy as np
import pandas as pd
import category_encoders as ce

from .utils import reduce_mem_usage

def prep_text(df):
    corpus = df['Name'].tolist()
    corpus = [re.sub('[:()!.-]', '', str(text)) for text in corpus]
    corpus = [text.lower() for text in corpus]

    remove_words = [' of ', ' the ', ' in ', ' a ', ' an ', ' vol', "'s"]
    for w in remove_words:
        corpus = [text.replace(w, ' ') for text in corpus]

    corpus = [text.replace(' iii', ' 3') for text in corpus]
    corpus = [text.replace(' ii', ' 2') for text in corpus]
    corpus = [text.replace(' iv', ' 4') for text in corpus]
    corpus = [text.replace(' v ', ' 5 ') for text in corpus]
    corpus = [text.replace(' i ', ' 1 ') for text in corpus]

    corpus = [text.replace('   ', ' ') for text in corpus]
    corpus = [text.replace('  ', ' ') for text in corpus]

    df['Name'] = corpus

    return df


def tfidf_vectorizer(df, tar_col, max_features=1000, ngram_range=(1, 1), n_components=50, type='svd'):
    """
    ソフト名をtfidfで変換し、SVDで圧縮
    """

    corpus = df[tar_col].tolist()

    # Tfidf
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)

    if type == 'svd':
        transformer = TruncatedSVD(n_components=n_components)
    elif type == 'lda':
        transformer = LatentDirichletAllocation(n_components=n_components)
    X = transformer.fit_transform(X)
    col_name = f"fe_tfidf_{ngram_range[0]}_{ngram_range[1]}_{type}_{tar_col}"
    out = pd.DataFrame(X, columns=[f"{col_name}_{i}" for i in range(n_components)])

    df = pd.concat([df, out], axis=1)

    return df


def idx_col_count_encode(df, cols, idx_col, n_components, type='svd'):
    """
    Publisherごとのカウントを集計し、次元削減する
    """

    for i, c in enumerate(cols):
        plat_pivot = df[[idx_col, c, 'Name']].pivot_table(index=idx_col, columns=c,values='Name', aggfunc='count', fill_value=0).reset_index().add_prefix(f'Count_{c}_')
        plat_pivot = plat_pivot.rename(columns={f'Count_{c}_{idx_col}': idx_col})
        if i == 0:
            temp = plat_pivot.copy()
        else:
            temp = pd.merge(temp, plat_pivot, on=[idx_col], how='left')


    temp.fillna(0, inplace=True)

    if type == 'svd':
        transformer = TruncatedSVD(n_components=n_components)
    elif type == 'lda':
        transformer = LatentDirichletAllocation(n_components=n_components)

    tar_cols = [c for c in temp.columns if c.startswith('Count_')]
    out = transformer.fit_transform(temp[tar_cols].values)

    out = pd.DataFrame(out, columns=[f"fe_{idx_col}_count_{type}_{i}" for i in range(n_components)])
    out[idx_col] = temp[idx_col]

    df = pd.merge(df, out, on=[idx_col], how='left')

    return df
