"""Shared bilingual controls for every page."""
import streamlit as st


def t(ja, en):
    return ja if st.session_state.get("language", "日本語") == "日本語" else en


def _save_language():
    st.session_state["language"] = st.session_state["language_selector"]


def sidebar():
    # Also hide the built-in list in servers started before config.toml changed.
    st.markdown(
        '<style>[data-testid="stSidebarNav"] {display: none;}</style>',
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("language", "日本語")
    st.session_state["language_selector"] = st.session_state["language"]
    st.sidebar.selectbox("言語 / Language", ["日本語", "English"],
                         key="language_selector", on_change=_save_language)
    with st.sidebar:
        st.page_link("main.py", label=t("ホーム", "Home"))
        st.page_link("pages/Classification from Draw.py", label=t("構造式から分類", "Classification from Draw"))
        st.page_link("pages/Classification from csv.py", label=t("CSVから分類", "Classification from CSV"))
        st.page_link("pages/Regression from Draw.py", label=t("構造式から回帰（改修中）", "Regression from Draw (Under maintenance)"))
        st.page_link("pages/Regression from csv.py", label=t("CSVから回帰（改修中）", "Regression from CSV (Under maintenance)"))


def display_results(df):
    if st.session_state.get("language", "日本語") != "日本語":
        return df
    df = df.copy()
    return df.rename(columns={"Prediction": "予測クラス", "Predicted_label": "予測ラベル",
        "probability_not_detected": "Not detectedの確率", "probability_low": "Lowの確率", "probability_high": "Highの確率"})



def class_caption():
    st.caption(t("分類：2 = High / 1 = Low / 0 = Not detected", "Classes: 2 = High / 1 = Low / 0 = Not detected"))
    st.markdown(t("**各クラスの定義**", "**Class definitions**"))
    st.markdown(t(
        "| クラス | 定義 |\n| --- | --- |\n| High | TEAC > 1 |\n| Low | TEAC ≤ 1 かつ IC50 ≤ 1000 µg/mL |\n| Not detected | IC50 > 1000 µg/mL |",
        "| Class | Definition |\n| --- | --- |\n| High | TEAC > 1 |\n| Low | TEAC ≤ 1 and IC50 ≤ 1000 µg/mL |\n| Not detected | IC50 > 1000 µg/mL |",
    ))
