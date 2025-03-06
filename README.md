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

### Regression from Draw (fr_Ar_OH = 1 の化合物のみ)
構造式を入力することで `pIC50` の予測値を取得できます。
  
### Regression from CSV (fr_Ar_OH = 1 の化合物のみ)
`SMILES` が記載された CSV ファイルをアップロードすることで、 `pIC50` の予測値を加えた CSV ファイルをダウンロードできます。
  
### Classification from Draw
構造式を入力することで `DPPH` 活性の判定予測が得られます。また、その予測に影響を与えた構造が表示されます。
  
### Classification from CSV
`SMILES` が記載された CSV ファイルをアップロードすることで、 `DPPH` 活性の判定予測を加えた CSV ファイルをダウンロードできます。
  
### Show FooDB Area (全ページ共通ボタン)
入力された化合物が `FooDB` の化合物範囲 (`fr_Ar_OH = 1`) のどこに位置するのかを可視化します。
