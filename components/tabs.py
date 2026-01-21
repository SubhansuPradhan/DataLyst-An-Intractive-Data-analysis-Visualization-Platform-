import pandas as pd
import streamlit as st

# Import each modular tab
from utils.visualization import chart_tab
from utils.feature_selection import feature_tab
from utils.prediction import prediction_tab
from utils.classification import classification_tab
from utils.forecasting import forecasting_tab
from utils.anomaly import anomaly_tab

def show_tabs(df, numeric_cols, cat_cols):

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Charts",
        "🔬 Feature Selection",
        "🔮 Prediction",
        "🧩 Classification",
        "⏳ Forecasting",
        "🚨 Anomaly Detection",
        "📄 Cleaned Data",
        "📏 Stats"
    ])

    with tab1:
        chart_tab(df, numeric_cols, cat_cols)

    with tab2:
        feature_tab(df)

    with tab3:
        prediction_tab(df, numeric_cols)

    with tab4:
        classification_tab(df, numeric_cols, cat_cols)

    with tab5:
        forecasting_tab(df, numeric_cols)

    with tab6:
        anomaly_tab(df)

    with tab7:
        st.dataframe(df, use_container_width=True)

    with tab8:
        stats_df = pd.DataFrame({
            "Count": df[numeric_cols].count(),
            "Mean": df[numeric_cols].mean(),
            "Median": df[numeric_cols].median(),
            "Mode": df[numeric_cols].mode().iloc[0],
            "Std Dev": df[numeric_cols].std(),
            "Variance": df[numeric_cols].var(),
            "Min": df[numeric_cols].min(),
            "Max": df[numeric_cols].max(),
            "IQR": df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
            "Skewness": df[numeric_cols].skew(),
            "Kurtosis": df[numeric_cols].kurtosis()
        })
        st.dataframe(stats_df)


