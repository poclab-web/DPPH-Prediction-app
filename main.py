import streamlit as st

st.title("メインページ")
st.write("ようこそ！メインページです。")

st.header('目次')
st.write("回帰予測を行うアプリケーションのページへ移動します。")
st.page_link("pages/Regression from Draw.py", label="▶ Regression from Draw", icon="🖋️")
st.page_link("pages/Regression from csv.py", label="▶ Regression from CSV", icon="📄")

st.write("二値分類を行うアプリケーションのページへ移動します。")
st.page_link("pages/Classification from Draw.py", label="▶ Classification from Draw", icon="🖋️")
st.page_link("pages/Classification from csv.py", label="▶ Classification from CSV", icon="📄")

st.header('READ ME')
st.write("このアプリケーションは、フェノール性水酸基を1つ持つ化合物の、DPPHフリーラジカル消去能を予測するためのアプリケーションです。")