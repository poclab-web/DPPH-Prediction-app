import pandas as pd
import streamlit as st
from rdkit.Chem import Fragments
from function.ui import sidebar, t, display_results, class_caption
from function.classification import load_classification_model, classification_features, prediction_frame

# ページの設定
sidebar()
st.title(t("CSVから分類", "Classification from CSV"))
st.subheader(t("CSVの書き方", "How to prepare your CSV"))
st.markdown(t(
    """1. **1行目に列名**を入れてください。構造の列名は **`SMILES`** を推奨します（`smiles`、`iso_SMILES`も使えます）。
2. **1行につき1化合物**のSMILESを入力してください。構造の列は1つだけ用意してください。
3. `name`などの識別用の列は任意で追加できます。これらの列も結果CSVに残ります。
4. **カンマ区切り・UTF-8形式のCSV**で保存してください。Excelでは「CSV UTF-8（コンマ区切り）」を選びます。""",
    """1. Put **column names in the first row**. Use **`SMILES`** for the structure column (`smiles` and `iso_SMILES` are also accepted).
2. Enter **one compound per row**, with one SMILES column in the file.
3. Optional columns such as `name` can identify compounds. These columns are retained in the results CSV.
4. Save as a **comma-separated UTF-8 CSV**. In Excel, choose “CSV UTF-8 (Comma delimited)”.""",
))
sample_csv = "name,SMILES\nphenol,Oc1ccccc1\ncatechol,Oc1ccccc1O\n"
st.code(sample_csv, language="text")
st.info(t(
    "SMILESの空欄や不正な構造がある場合、またはフェノール構造を含まない化合物がある場合は、ファイル全体の予測を停止します。化合物名だけでは予測できません。",
    "Prediction stops for the entire file if any SMILES is missing or invalid, or if any compound has no phenol structure. Compound names alone cannot be used for prediction.",
))
st.caption(t(
    "値にカンマを含む場合は、その値を二重引用符で囲んでください。予測結果にはクラス（0=低、1=中、2=高）と各クラスの確率が追加されます。ダウンロードする結果の列名・ラベルは英語です。",
    'Enclose values containing commas in double quotes. Results add a class (0=Low, 1=Middle, 2=High) and its class probabilities. Downloaded column names and labels are in English.',
))
st.download_button(t("サンプルCSVをダウンロード", "Download a sample CSV"),
                   sample_csv.encode("utf-8-sig"), "sample.csv", mime="text/csv")
smiles_file = st.file_uploader(t("SMILESを含むCSVをアップロードしてください。", "Upload a CSV containing SMILES."), type='csv')

load_model = st.cache_resource(load_classification_model)
class_caption()

if st.button(t("アップロードしたCSVを予測", "Predict uploaded CSV")):
    if smiles_file is None:
        st.error(t("CSVファイルをアップロードしてください。", "Please upload a CSV file."))
        st.stop()
    try:
        df = pd.read_csv(smiles_file)
        smiles_column = next((c for c in ("SMILES", "smiles", "iso_SMILES") if c in df.columns), None)
        if smiles_column is None:
            raise ValueError(t("SMILES、smiles、iso_SMILESのいずれかの列が必要です。", "CSV requires a SMILES, smiles or iso_SMILES column."))
        features, molecules, _ = classification_features(df[smiles_column])
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(t("CSVまたはSMILESを確認してください。SMILES列の欠損や不正な入力がないか確認してください。", str(exc)))
        st.stop()

    if any(Fragments.fr_Ar_OH(mol) == 0 or Fragments.fr_phenol(mol) == 0 for mol in molecules):
        st.error(t("フェノール構造を含まない行があります。", "Some rows do not contain phenol."))
        st.stop()

    result = prediction_frame(load_model(), features)
    for column in result:
        df[column] = result[column].to_numpy()
    st.dataframe(display_results(df))
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(t("予測結果をダウンロード", "Download predictions"), csv, 'Prediction_Classification.csv', mime="text/csv")
