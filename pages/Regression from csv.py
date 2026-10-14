import streamlit as st
from function.ui import sidebar, t

sidebar()
st.title(t("CSVから回帰", "Regression from CSV"))
st.info(t("現在改修中です。回帰予測は一時的にご利用いただけません。", "Under maintenance. Regression predictions are temporarily unavailable."))
st.stop()

# Legacy regression implementation retained for the planned renovation.
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools
from mordred import Calculator, descriptors
import xgboost as xgb
from function.generate import generate_fingerprint_df, add_phenol_check, add_descriptor
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
