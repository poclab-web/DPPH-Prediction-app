import base64
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit as st
import lightgbm as lgb
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools
from rdkit.Chem.Draw import rdMolDraw2D
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from function.generate import generate_fingerprint_df, visualize_morgan_shap, mol_to_base64


# ページの設定
st.markdown("# Classification from Draw")
st.text('draw chemical structure or put SMILES')
search_smiles = st_ketcher("CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C")
st.text('SMILES')
st.code(search_smiles)
input_mol = Chem.MolFromSmiles(search_smiles)

# 予測
if st.button("Start prediction"):
    # フィンガープリント生成
    df = pd.DataFrame([input_mol], columns=["ROMol"])
    df["SMILES"] = Chem.MolToSmiles(input_mol)
    fp_df = generate_fingerprint_df(df)

    # pickle ファイルからモデルを読み込む
    with open("Classification_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    
    # モデルを使って予測する
    prediction = loaded_model.predict(fp_df)
    st.write(True if prediction[0] == 1 else False)    

    explainer = shap.Explainer(loaded_model)
    shap_values = explainer(fp_df)

    # SHAP値のプロット
    svg = visualize_morgan_shap(input_mol, shap_values.values[0])
    st.image(svg)

# 化合物空間の表示
if st.button("Show FooDB Area"):

    # FooDB のデータを読み込む
    df_FooDB = pd.read_csv("chem_area_fr_Ar_OH_1_FooDB.csv")
    df_FooDB = df_FooDB[["name", "iso_SMILES", "mw", "MolLogP"]]
    df_FooDB["Data_source"] = "FooDB"
    PandasTools.AddMoleculeColumnToFrame(df_FooDB, "iso_SMILES")

    # 分子画像を Base64 に変換
    df_FooDB["base64image"] = df_FooDB["ROMol"].apply(mol_to_base64)

    # import smilesの情報を取得
    input_mol_mw = Descriptors.MolWt(input_mol)
    input_mol_logp = Descriptors.MolLogP(input_mol)

    # データフレームの作成
    df_test = pd.DataFrame([["test_compound", search_smiles, input_mol_mw, input_mol_logp, "Test"]], columns=["name", "iso_SMILES", "mw", "MolLogP", "Data_source"])
    PandasTools.AddMoleculeColumnToFrame(df_test, "iso_SMILES")
    df_test["base64image"] = df_test["ROMol"].apply(mol_to_base64)

    # データフレームの結合
    df_FooDB_2 = pd.concat([df_FooDB, df_test], axis=0)

    # Bokeh用データソース
    source = ColumnDataSource(df_FooDB_2[["iso_SMILES", "name", "MolLogP", "mw", "base64image", "Data_source"]])

    # 図を作成
    p = figure(
        title="mw vs. MolLogP (fr_Ar_OH = 1)",
        x_axis_label="mw",
        y_axis_label="MolLogP",
        tools="hover, pan, box_zoom, reset, wheel_zoom",
        match_aspect=False,  # 縦横比を固定
    )

    # データポイントをプロット（色分け）
    p.scatter(
        x="mw",
        y="MolLogP",
        source=source,
        size=8,
        line_color=None,
        fill_color=factor_cmap(
            "Data_source",
            palette=["blue", "gray"],  # Test は青色、FooDB は灰色
            factors=["Test", "FooDB"],
        ),
        legend_field="Data_source",
        alpha=0.6,
    )

    # ツールチップを設定
    hover = HoverTool()
    hover.tooltips = """
        <div>
            <div>
                <span style="font-size: 12px; color: #2f7cf7;">Name:</span>
                <span style="font-size: 12px;">@name</span>
            </div>
            <div>
                <img src="data:image/png;base64,@base64image" style="max-width: 200px;"/>
            </div>
            <div>
                <span style="font-size: 12px; color: #2f7cf7;">Source:</span>
                <span style="font-size: 12px;">@Data_source</span>
            </div>
        </div>
    """
    p.add_tools(hover)

    # 凡例の位置と設定
    p.legend.location = "top_right"  # 右上に配置
    p.legend.background_fill_alpha = 0.5  # 背景の透明度を設定

    # Streamlit で Bokeh を表示
    st.bokeh_chart(p)