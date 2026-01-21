import streamlit as st
import pandas as pd
import plotly.express as px

# Component Imports
from components.sidebar import load_sidebar
from components.header import load_header
from components.tabs import show_tabs

# Page Config
st.set_page_config(page_title="DataLyst", layout="wide")

# Background & UI Styling
background_image_url = "https://i.postimg.cc/XJShYJMd/unwatermark-stock-footage-high-tech-data-interface-with-graphs-charts-and-futuristic-visuals-perfect.gif"

st.markdown(f"""
<style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid white;
    }}
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0.0);
        box-shadow: none;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar → returns df, numeric_cols, cat_cols
df, numeric_cols, cat_cols = load_sidebar()

# Header
load_header()

# If file uploaded → Load Tabs
if df is not None:
    show_tabs(df, numeric_cols, cat_cols)
else:
    st.info("Upload a dataset to begin using DataLyst.")
