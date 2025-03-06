import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools
from mordred import Calculator, descriptors
import xgboost as xgb
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from function.generate import generate_fingerprint_df, mol_to_base64, add_phenol_check, add_descriptor
from function.generate import calculate_o1_charge_for_phenol_derivatives, calculate_descriptors


# ページの設定
st.markdown("# DPPH Regression from CSV")
smiles_file = st.file_uploader('smilesのcsvファイルをアップロードしてください。', type = 'csv')
with open('smiles_example.csv', 'rb')as file:
    st.download_button('Download a sample file', file, 'sample.csv')

if st.button('Predict_uploaded_file'):
    df = pd.read_csv(smiles_file)
    errors = []
    keys = []
    smiless = []

    if 'SMILES' in df.columns:
        smiless = df['SMILES']
    elif 'smiles' in df.columns:
        smiless = df['smiles']
        df = df.rename(columns={'smiles': 'SMILES'})
    else:
        for column in df.columns:
            for n, smiles in enumerate(df[column]):
                try:
                    smiless.append(smiles)
                    keys.append(n)
                except Exception as e:
                    errors.append((smiles, str(e)))

    if errors:
        st.error(f"Errors occurred while processing the following SMILES: {errors}")

    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="SMILES")

    df = add_phenol_check(df)

    if (df["fr_Ar_OH"] != 1).any() or (df["fr_phenol"] != 1).any():
        st.write("Some rows do not contain phenol.")
        st.stop()

    else:
        fp_df = generate_fingerprint_df(df)

        dtest_HOMO = xgb.DMatrix(fp_df)

        # pickle ファイルからモデルを読み込む
        with open("xgb_HOMO_fp_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # モデルを使って予測する
        HOMO = loaded_model.predict(dtest_HOMO)
        df['HOMO'] = HOMO

        # 4. 分子オブジェクト列から記述子を計算
        mols = df["ROMol"].tolist()
        descriptors_list = [calculate_descriptors(mol) for mol in mols]

        # 5. 記述子をデータフレーム化
        df_descriptors = pd.DataFrame(descriptors_list)

        # 6. 記述子を元データフレームに結合
        df = pd.concat([df, df_descriptors], axis=1)

        df = calculate_o1_charge_for_phenol_derivatives(df)

        df = df.apply(pd.to_numeric, errors='coerce')  # 数値変換できないものは NaN になる
        df = df.fillna(0)  # NaN を 0 に変換

        df = df[['HOMO', 'ABC', 'ABCGG', 'SpAbs_A', 'SpMax_A', 'SpMAD_A', 'AATS0dv', 'AATS3dv', 'AATS1p', 'AATS0i', 'ATSC3c', 'ATSC5c', 'ATSC4dv', 'ATSC7dv', 'ATSC7d', 'GATS1s', 'o_1']]

        dtest_pIC50 = xgb.DMatrix(df)

        # pickle ファイルからモデルを読み込む
        with open("xgb_pIC50_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # モデルを使って予測する
        result = loaded_model.predict(dtest_pIC50)
        df['result'] = result
        
        st.dataframe(df)
        csv = df.to_csv().encode("utf-8")
        st.download_button('Download the Prediction', csv, 'Prediction.csv')


# 化合物空間の表示
if st.button("Show FooDB Area"):
    df = pd.read_csv(smiles_file)
    errors = []
    keys = []
    smiless = []

    # FooDB のデータを読み込む
    df_FooDB = pd.read_csv("chem_area_fr_Ar_OH_1_FooDB.csv")
    df_FooDB = df_FooDB[["name", "iso_SMILES", "mw", "MolLogP"]]
    df_FooDB["Data_source"] = "FooDB"
    PandasTools.AddMoleculeColumnToFrame(df_FooDB, "iso_SMILES")

    # 分子画像を Base64 に変換
    df_FooDB["base64image"] = df_FooDB["ROMol"].apply(mol_to_base64)

    if 'SMILES' in df.columns:
        smiless = df['SMILES']
    elif 'smiles' in df.columns:
        smiless = df['smiles']
        df = df.rename(columns={'smiles': 'SMILES'})
    else:
        for column in df.columns:
            for n, smiles in enumerate(df[column]):
                try:
                    smiless.append(smiles)
                    keys.append(n)
                except Exception as e:
                    errors.append((smiles, str(e)))

    if errors:
        st.error(f"Errors occurred while processing the following SMILES: {errors}")

    # データフレームの作成
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="SMILES")
    df = add_descriptor(df)
    df["base64image"] = df["ROMol"].apply(mol_to_base64)
    df = df.rename(columns={'SMILES': 'iso_SMILES'})
    df["Data_source"] = "Test"

    df_FooDB = df_FooDB.rename(columns={'mw': 'MolWt'})

    df = df[["iso_SMILES", "MolLogP", "MolWt", "base64image", "Data_source"]]
    df_FooDB = df_FooDB[["iso_SMILES", "MolLogP", "MolWt", "base64image", "Data_source"]]

    # データフレームの結合
    df_FooDB_2 = pd.concat([df_FooDB, df], axis=0)

    # Bokeh用データソース
    source = ColumnDataSource(df_FooDB_2[["iso_SMILES", "MolLogP", "MolWt", "base64image", "Data_source"]])

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
        x="MolWt",
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