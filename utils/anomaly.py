"""
Anomaly detection tab: IsolationForest and DBSCAN visualizations with detailed summaries.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def anomaly_tab(df: pd.DataFrame):
    st.markdown("## Anomaly Detection")

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric:
        st.info("No numeric columns available for anomaly detection.")
        return

    features = st.multiselect("Numeric features", numeric, default=numeric[:3], key="ano_features")
    method = st.selectbox("Method", ["Isolation Forest", "DBSCAN"], key="ano_method")

    if st.button("Run Anomaly Detection"):
        X = df[features].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        if method == "Isolation Forest":
            n_estimators = int(st.number_input("n_estimators", min_value=10, max_value=1000, value=100))
            iso = IsolationForest(n_estimators=n_estimators, contamination='auto', random_state=42)
            preds = iso.fit_predict(Xs)
            df_res = X.copy()
            df_res["Anomaly"] = np.where(preds == -1, "Anomaly", "Normal")
            st.success(f"Detected {(df_res['Anomaly']=='Anomaly').sum()} anomalies out of {len(df_res)} samples.")

            # ------------------ Scatter Matrix ------------------
            fig = px.scatter_matrix(
                df_res,
                dimensions=features,
                color="Anomaly",
                title="Isolation Forest: Anomaly Scatter Matrix",
                width=1200,
                height=800,
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # ------------------ Summary ------------------
            st.subheader("Isolation Forest Summary")
            counts = df_res["Anomaly"].value_counts()
            for label, count in counts.items():
                st.markdown(f"- **{label}**: `{count}` samples")

        else:  # DBSCAN
            eps = float(st.number_input("eps", min_value=0.01, max_value=10.0, value=0.5, step=0.01))
            min_samples = int(st.number_input("min_samples", min_value=1, max_value=50, value=5))
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(Xs)
            df_res = X.copy()
            df_res["Cluster"] = labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            st.success(f"Found {n_clusters} clusters; noise points: {n_noise}")

            # ------------------ Scatter Matrix ------------------
            fig = px.scatter_matrix(
                df_res,
                dimensions=features,
                color="Cluster",
                title="DBSCAN: Cluster Scatter Matrix",
                width=1200,
                height=800,
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # ------------------ Summary ------------------
            st.subheader("DBSCAN Summary")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                label = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                st.markdown(f"- **{label}**: `{count}` samples")

