import streamlit as st
import pandas as pd

def load_sidebar():
    st.sidebar.header("🔧 Controls")

    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file:", type=["csv", "xlsx"])

    if uploaded_file is None:
        return None, None, None

    # load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # limit rows
    if len(df) > 50000:
        df = df.head(50000)

    st.sidebar.subheader("🧹 Cleaning Options")
    cleaning_option = st.sidebar.radio("Handle Missing Values:", ["Drop rows", "Fill with Mean", "Fill with Zero"])

    if cleaning_option == "Drop rows":
        df = df.dropna()
    elif cleaning_option == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    else:
        df = df.fillna(0)

    selected_columns = st.sidebar.multiselect(
        "📌 Select columns to keep:",
        df.columns.tolist(),
        default=df.columns.tolist()
    )

    df = df[selected_columns]

    # row range slider
    max_rows = len(df)
    row_range = st.sidebar.slider("Select Row Range:", 1, max_rows, (1, min(50, max_rows)))
    df = df.iloc[row_range[0]-1: row_range[1]]

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return df, numeric_cols, cat_cols
