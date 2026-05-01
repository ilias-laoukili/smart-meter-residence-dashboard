"""
Real Estate Analysis Dashboard — Streamlit entry point.

Run locally:
    streamlit run src/front_end/app.py

Deploy to Streamlit Cloud:
    Main file path: src/front_end/app.py
    (run scripts/precompute.py once to generate data/processed/features.parquet
     and models/saved_models/ before pushing to the repository)

Navigation is handled via ``st.navigation`` (Streamlit >= 1.36).
"""

import streamlit as st

st.set_page_config(
    page_title="Real Estate Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
home_page       = st.Page("pages/home.py",            title="Home & KPIs",      icon="📊", default=True)
classification_page = st.Page("pages/classification.py", title="Classification", icon="🏷️")
regression_page = st.Page("pages/regression.py",      title="Regression",        icon="📈")
generation_page = st.Page("pages/data_generation.py", title="Data Generation",   icon="🧬")
explorer_page   = st.Page("pages/explorer.py",        title="Property Explorer", icon="🔍")

nav = st.navigation(
    {
        "Overview":       [home_page],
        "Modelling":      [classification_page, regression_page],
        "Synthetic Data": [generation_page],
        "Exploration":    [explorer_page],
    }
)
with st.sidebar:
    st.markdown("---")
    st.caption("Real Estate Analysis v2.0")
    st.caption("Smart-meter data · 500 properties · Nov 2023 – Oct 2024")
    st.caption("Labels: professor's ground-truth annotation")

nav.run()
