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
    if "Predicted_label" in df:
        df["Predicted_label"] = df["Predicted_label"].map({"Low": "低", "Middle": "中", "High": "高"})
    return df.rename(columns={"Prediction": "予測クラス", "Predicted_label": "予測ラベル",
        "probability_low": "低の確率", "probability_middle": "中の確率", "probability_high": "高の確率"})


def class_caption():
    st.caption(t("分類：0 = 低 / 1 = 中 / 2 = 高", "Classes: 0 = Low / 1 = Middle / 2 = High"))
