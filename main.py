import streamlit as st
from function.ui import sidebar, t

sidebar()
st.title("DPPH Analyzer")
st.write(t("化合物のDPPHフリーラジカル消去能を予測します。", "Predict DPPH free radical scavenging activity of compounds."))
st.header(t("分類予測", "Classification"))
st.write(t("構造式またはSMILESから、活性を低・中・高の3クラスに分類します。", "Classify activity as Low, Middle or High from a chemical structure or SMILES."))
st.page_link("pages/Classification from Draw.py", label=t("構造式から分類", "Classification from Draw"), icon="🖋️")
st.page_link("pages/Classification from csv.py", label=t("CSVから分類", "Classification from CSV"), icon="📄")
st.header(t("回帰予測", "Regression"))
st.info(t("回帰モデルのページは現在改修中です。", "Regression pages are currently under maintenance."))
st.page_link("pages/Regression from Draw.py", label=t("構造式から回帰（改修中）", "Regression from Draw (Under maintenance)"), disabled=True)
st.page_link("pages/Regression from csv.py", label=t("CSVから回帰（改修中）", "Regression from CSV (Under maintenance)"), disabled=True)
