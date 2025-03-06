import base64
import pickle
import pandas as pd
import streamlit as st
import lightgbm as lgb
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from function.generate import generate_fingerprint_df, mol_to_base64, add_phenol_check
from function.generate import calculate_o1_charge_for_phenol_derivatives, calculate_descriptors
import xgboost as xgb


# ページの設定
st.markdown("# DPPH Regressiom from Draw")
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
    st.write(result)


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