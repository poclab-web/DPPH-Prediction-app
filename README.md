# DPPH Analyzer

## Streamlit Community Cloudへの公開

クラウドでは `requirements.txt` と `packages.txt` を使用します。
デプロイ時の **Advanced settings → Python version は3.11** にしてください。
現在の依存バージョンはPython 3.12以降を対象としていません。

- Repository：公開先のリポジトリ
- Branch：`master`
- Main file path：`main.py`
- `packages.txt`：LightGBM用のOpenMPランタイム（`libgomp1`）とRDKit描画用のCairo・X11ライブラリ（`libcairo2`、`libxrender1`、`libxext6`）

`environment.yml` があるとcondaが優先されるため、ローカル用ファイルは
`environment.local.yml` という名前にしています。改名・追加をまとめてGitHubに反映してください。
既存アプリのPythonバージョンを変更する場合は、URL・Secretsなどの設定を控え、
アプリを削除してPython 3.11を選択して再デプロイします。

## ローカル環境構築
### 必要な環境
- `Python 3.8.20`

### モジュールのインストール
以下のコマンドを実行して必要なモジュールをインストールしてください。
```sh
conda env create -f environment.local.yml
conda activate streamlit_DPPH
```
※ Windows環境以外では正常に動作しない可能性があります。

### 環境ファイルについて
`environment.local.yml` は現在のStreamlit分類アプリの実行用です。
主要パッケージは動作確認済みのバージョンを指定し、間接依存はconda/pipに解決させます。
OS固有のビルド番号と個人環境の`prefix`は含めていません。

- UI：Streamlit、streamlit-ketcher
- 特徴量・推論：NumPy、pandas、RDKit、LightGBM、scikit-learn、SciPy、joblib

改修中の回帰用のXGBoost・Mordred、削除済みのFooDB表示用のBokeh、
学習・ノートブック用のJupyter・SHAPなどは含めていません。
回帰機能の再開やノートブック実行時には別途依存関係の追加が必要です。
分類画面の構造寄与表示はLightGBM内蔵のTree SHAPを使うため、SHAPパッケージは不要です。

このファイルは全依存を固定したロックファイルではありません。
既存の環境はファイルを編集するだけでは変更されません。

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
