import streamlit as st
from function.ui import sidebar, t

sidebar()
st.title(t("構造式から回帰", "Regression from Draw"))
st.info(t("現在改修中です。回帰予測は一時的にご利用いただけません。", "Under maintenance. Regression predictions are temporarily unavailable."))
st.stop()

# Legacy regression implementation retained for the planned renovation.
import pickle
import pandas as pd
import streamlit as st
import lightgbm as lgb
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from function.generate import generate_fingerprint_df, add_phenol_check
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
