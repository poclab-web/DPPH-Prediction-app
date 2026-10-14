import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
from function.ui import sidebar, t, display_results, class_caption
from function.classification import (load_classification_model, classification_features,
    prediction_frame, project_shap_to_molecule)


# ページの設定
sidebar()
st.title(t("構造式から分類", "Classification from Draw"))
st.write(t("構造式を描くか、SMILESを入力してください。", "Draw a chemical structure or enter SMILES."))
search_smiles = st_ketcher("CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C")
st.text('SMILES')
st.code(search_smiles)

# 予測
load_model = st.cache_resource(load_classification_model)
class_caption()

if st.button(t("予測を開始", "Start prediction")):
    try:
        features, molecules, bitinfos = classification_features([search_smiles])
    except ValueError as exc:
        st.error(t("SMILESが空か、不正な形式です。入力を確認してください。", str(exc)))
        st.stop()
    loaded_model = load_model()
    result = prediction_frame(loaded_model, features)
    st.dataframe(display_results(result), hide_index=True)
    st.caption(t("構造の寄与（PAVA補正前）：赤は正の寄与、青は負の寄与を示します。", "SHAP before PAVA calibration: red = positive, blue = negative contribution."))
    for title, tree_model in [
        (t("Low・Highへの寄与", "Contribution toward Low / High"), loaded_model.model.model_above_low_),
        (t("Highへの寄与", "Contribution toward High"), loaded_model.model.model_high_),
    ]:
        st.write(title)
        # Native LightGBM Tree SHAP; the final column is the expected value.
        values = tree_model.booster_.predict(features, pred_contrib=True)[:, :-1]
        _, svg = project_shap_to_molecule(
            0, values, molecules, bitinfos, pd.DataFrame(), show=False,
        )
        st.image(svg)
