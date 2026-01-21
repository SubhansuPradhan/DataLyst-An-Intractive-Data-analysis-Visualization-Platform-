"""
Visualization tab functions.
Uses plotly.express and Streamlit layout controls.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from typing import List, Optional

def chart_tab(df: pd.DataFrame,
              numeric_cols: Optional[List[str]] = None,
              cat_cols: Optional[List[str]] = None,
              color_sequence=None):
    """
    Renders charts. Arguments:
      - df: DataFrame slice to visualize
      - numeric_cols: list of numeric columns
      - cat_cols: list of categorical columns
      - color_sequence: optional Plotly color sequence
    """
    if df is None:
        st.info("No data loaded.")
        return

    if color_sequence is None:
        import plotly.express as _px
        color_sequence = _px.colors.qualitative.Plotly

    # --- Histogram ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/histogram.png", width=60)
    with col2:
        st.markdown("<h3>Histogram</h3>", unsafe_allow_html=True)
    if numeric_cols:
        col_hist = st.selectbox("Pick numeric column for histogram", numeric_cols, key="hist")
        fig_hist = px.histogram(df, x=col_hist, color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No numeric columns for histogram.")

    # --- Bar ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/bar.png", width=60)
    with col2:
        st.markdown("<h3>Bar Chart</h3>", unsafe_allow_html=True)
    if cat_cols and numeric_cols:
        col_cat = st.selectbox("Categorical column (X)", cat_cols, key="bar_x")
        col_num = st.selectbox("Numeric column (Y)", numeric_cols, key="bar_y")
        fig_bar = px.bar(df, x=col_cat, y=col_num, color=col_cat, color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Bar chart requires at least one categorical and one numeric column.")

    # --- Donut/Pie ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/donut.png", width=60)
    with col2:
        st.markdown("<h3>Donut / Pie</h3>", unsafe_allow_html=True)
    if cat_cols and numeric_cols:
        donut_labels = st.selectbox("Categorical labels (donut)", cat_cols, key="donut_labels")
        donut_values = st.selectbox("Numeric values (donut)", numeric_cols, key="donut_values")
        fig_donut = px.pie(df, names=donut_labels, values=donut_values, hole=0.4,
                           color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Donut/Pie requires categorical + numeric column.")

    # --- Scatter ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/scatter.png", width=60)
    with col2:
        st.markdown("<h3>Scatter Plot</h3>", unsafe_allow_html=True)
    if numeric_cols:
        x_col = st.selectbox("X column", df.columns.tolist(), key="scatter_x")
        y_col = st.selectbox("Y column (numeric)", numeric_cols, key="scatter_y")
        color_by = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist(), key="scatter_color")
        fig_scat = px.scatter(df, x=x_col, y=y_col,
                              color=(None if color_by == "None" else color_by),
                              color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info("No numeric columns for scatter.")

    # --- Box ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/box.png", width=60)
    with col2:
        st.markdown("<h3>Box Plot</h3>", unsafe_allow_html=True)
    if numeric_cols:
        col_box = st.selectbox("Numeric column for box", numeric_cols, key="box_col")
        cat_for_box = st.selectbox("Group by (optional)", ["None"] + (cat_cols or []), key="box_group")
        fig_box = px.box(df, x=(None if cat_for_box == "None" else cat_for_box), y=col_box,
                         points="all", color=(None if cat_for_box == "None" else cat_for_box),
                         color_discrete_sequence=color_sequence)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns for box plot.")

    # --- Heatmap / Correlation ---
    st.markdown("---")
    col1, col2 = st.columns([1, 20])
    with col1:
        st.image("./assets/image/heat.png", width=60)
    with col2:
        st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
    if numeric_cols and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Heatmap requires at least 2 numeric columns.")
