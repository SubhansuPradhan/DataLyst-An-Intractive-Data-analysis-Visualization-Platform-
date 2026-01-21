"""
Feature selection tab: Model-based RFE + importance.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

def feature_tab(df: pd.DataFrame):
    st.markdown("## Feature Selection (RFE + Importance)")

    cols = df.columns.tolist()
    target = st.selectbox("Target column", cols, key="fs_target")
    candidates = st.multiselect("Candidate features", [c for c in cols if c != target], default=[c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target], key="fs_candidates")
    method = st.selectbox("Estimator", ["Linear Regression", "Decision Tree", "Random Forest"], key="fs_est")

    if st.button("Run Feature Selection"):
        if not candidates:
            st.error("Select candidate features.")
            return
        df_fs = df[candidates + [target]].dropna()
        X = df_fs[candidates]
        y = df_fs[target]

        if method == "Linear Regression":
            estimator = LinearRegression()
        elif method == "Decision Tree":
            estimator = DecisionTreeRegressor(random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

        n_select = min(max(1, min(5, len(candidates))), len(candidates))
        rfe = RFE(estimator=estimator, n_features_to_select=n_select)
        rfe.fit(X, y)
        selected = [f for f, s in zip(candidates, rfe.support_) if s]

        try:
            estimator.fit(X, y)
            importances = getattr(estimator, "feature_importances_", np.abs(getattr(estimator, "coef_", np.zeros(len(candidates)))))
        except Exception:
            importances = np.zeros(len(candidates))

        df_imp = pd.DataFrame({
            "Feature": candidates,
            "Selected": rfe.support_,
            "Rank": rfe.ranking_,
            "Importance": np.round(importances, 4)
        }).sort_values(["Selected", "Importance"], ascending=[False, False])

        st.success(f"Selected: {', '.join(selected)}")
        st.dataframe(df_imp)
        fig = px.bar(df_imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
