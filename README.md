# DPPH Analyzer

## 環境構築
### 必要な環境
- `Python 3.8.20`

### モジュールのインストール
以下のコマンドを実行して必要なモジュールをインストールしてください。
```sh
conda env create -f environment.yml
```
※ Windows環境以外では正常に動作しない可能性があります。

### 必要なモジュール一覧 (conda)
```sh
numpy == 1.24.3
pandas == 2.0.3
streamlit == 1.38.0
streamlit-ketcher == 0.0.1 (pip)
rdkit == 2023.03.3
mordred == 1.2.0
lightgbm == 4.5.0
xgboost == 2.1.1
shap == 0.42.1
bokeh == 2.4.3
```

## 使い方
環境をアクティブにした後、以下のコマンドを実行するとローカル環境でアプリが起動します。
```sh
streamlit run main.py
```

## 機能の説明

分類ページは `best_ordinal_lgbm_pava.pkl` を使用します。
`ordinal_lightgbm_classification.ipynb` と同じ半径3・2048ビットのMorgan fingerprint、
累積二値LightGBM、PAVA確率補正で推論します。再学習は不要です。
結果は `0 = Low / 1 = Middle / 2 = High` と各クラスの確率を表示・CSV出力します。
ラベル名はノートブックの予測CSV出力に合わせています（一部の図のラベルとは異なります）。
Drawの構造寄与表示は、PAVA補正前の2つの分類器に対するSHAPです。
独自クラスと読み込み処理は `function/classification.py` にあります。

### 回帰ページ（改修中）
回帰予測は一時停止しています。直接ページを開いても改修中の案内のみ表示します。

### 表示言語
サイドバーの「言語 / Language」で日本語・英語を切り替えられます。選択はページ移動後も維持されます。
予測CSVの列名・ラベルは互換性のため英語で保存します。

### Classification from Draw
構造式を入力することで `DPPH` 活性の判定予測が得られます。また、その予測に影響を与えた構造が表示されます。
  
### Classification from CSV
`SMILES` が記載された CSV ファイルをアップロードすることで、 `DPPH` 活性の判定予測を加えた CSV ファイルをダウンロードできます。
