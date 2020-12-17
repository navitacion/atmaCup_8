# atmaCup_8

atmaCup #8のソースコード


## 環境構築

Comet_MLでプロジェクトを作成し、発行されるapi_key, project_nameを、config.ymlの下記該当箇所で置換

```
exp:
  api_key: **YOUR_API_KEY**
  project_name: **YOUR_PROJECT_NAME**
```

Dockerコンテナをビルド

```
docker build -t atma_env .
```


## モデルの学習＆推論

ビルドしたコンテナを起動する

```
docker run -it --rm --gpus all -v $(pwd):/workspace atma_env bash
```

dockerコンテナ内で下記コマンドを実行

```
python train.py
```

コマンドライン引数で条件を指定して学習

```
# fold数を7, 学習率を0.01で実行
python train.py data.n_splits=7 train.lr=0.01
```

予測結果やSubmitファイルはComet_MLにアップロードされます。
