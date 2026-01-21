"""
Optimized Classification Module
Theme: Blue–Purple Modern Theme
Adaptive diagrams:
- KNN: 2D decision boundary / 3D scatter (PCA fallback)
- Decision Tree: interactive tree graph (Plotly)
Includes: Model summaries (KNN + Decision Tree)
"""

from io import BytesIO
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree, export_text

# --------------------- Safe thresholds (Option B: Medium) ---------------------
MAX_ROWS = 100_000
MAX_COLS = 35
MAX_CLASSES = 15
MAX_FEATURES_FOR_PLOTTING = 50  # guard for plotting

# --------------------- Safety Checks ---------------------
def dataset_too_large(df):
    """Block classification for extremely large datasets."""
    if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
        return True
    return False

def too_many_classes(y):
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) > MAX_CLASSES:
        return True
    return False

def too_many_features(features):
    if len(features) > MAX_FEATURES_FOR_PLOTTING:
        return True
    return False

# --------------------- Safe train/test split ---------------------
def safe_train_test_split(X_encoded, y, test_size=0.2, random_state=42):
    if X_encoded is None or y is None:
        st.warning("Empty features or target provided.")
        return None, None, None, None

    if len(y) < 2:
        st.warning("Not enough samples to split the dataset (need at least 2).")
        return None, None, None, None

    class_counts = pd.Series(y).value_counts()
    stratify_param = y if class_counts.min() >= 2 else None

    try:
        return train_test_split(X_encoded, y, test_size=test_size,
                                random_state=random_state, stratify=stratify_param)
    except Exception as e:
        st.error(f"Train/test split failed: {e}")
        return None, None, None, None

# --------------------- Convert Matplotlib figure to PNG ---------------------
def mpl_to_streamlit(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return buf

# --------------------- KNN Summary ---------------------
def knn_summary(model, X_train, y_train, X_test, y_test, class_names):
    try:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
    except Exception:
        st.warning("KNN summary: prediction failed.")
        return

    st.subheader("📘 KNN Summary")
    st.markdown(f"**Accuracy:** `{acc*100:.2f}%`")
    st.markdown(f"- **Distance metric:** `{model.metric}`")
    st.markdown(f"- **k (Neighbors):** `{model.n_neighbors}`")

    st.markdown("### Class Distribution (Train)")
    dist = pd.Series(y_train).value_counts().sort_index()
    for idx, count in dist.items():
        st.markdown(f"- **{class_names[idx]}** → `{count}` samples")

    st.markdown("### Dataset Info")
    st.markdown(f"- Train samples: `{len(X_train)}`")
    st.markdown(f"- Test samples: `{len(X_test)}`")

# --------------------- Decision Tree Summary ---------------------
def decision_tree_summary(model, X_train, y_train, X_test, y_test, class_names):
    try:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
    except Exception:
        st.warning("Decision Tree summary: prediction failed.")
        return

    st.subheader("🌳 Decision Tree Summary")
    st.markdown(f"**Accuracy:** `{acc*100:.2f}%`")

    depth = model.get_depth()
    leaves = model.get_n_leaves()
    st.markdown(f"- **Tree Depth:** `{depth}`")
    st.markdown(f"- **Leaves:** `{leaves}`")

    st.markdown("### Feature Importances")
    fi = model.feature_importances_
    names = list(X_train.columns)
    if np.sum(fi) == 0:
        st.info("Tree found no meaningful splits.")
    else:
        st.table(
            pd.DataFrame({"Feature": names, "Importance": fi})
              .sort_values("Importance", ascending=False)
        )

    st.markdown("### Class Distribution (Train)")
    dist = pd.Series(y_train).value_counts().sort_index()
    for idx, count in dist.items():
        st.markdown(f"- **{class_names[idx]}** → `{count}` samples")

    st.markdown("### Dataset Info")
    st.markdown(f"- Train samples: `{len(X_train)}`")
    st.markdown(f"- Test samples: `{len(X_test)}`")

# --------------------- Main Classification Function ---------------------
def classification_tab(df: pd.DataFrame, numeric_cols, cat_cols):
    st.markdown("## Classification — KNN & Decision Tree")

    # ------------------ Safety checks ------------------
    if dataset_too_large(df):
        st.error(f"❌ Dataset too large (> {MAX_ROWS} rows or > {MAX_COLS} columns).")
        st.info("Reduce dataset size or select a smaller subset.")
        return

    if df is None or df.empty:
        st.warning("No dataset loaded.")
        return

    if not cat_cols:
        st.info("No categorical target found.")
        return

    # ------------------ User Inputs ------------------
    target = st.selectbox("Select target column (categorical)", cat_cols)
    features = st.multiselect("Select feature columns (X)",
                              [c for c in df.columns if c != target],
                              default=numeric_cols)

    if not target or not features:
        st.warning("Select target and features.")
        return

    # ------------------ Feature threshold ------------------
    if too_many_features(features):
        st.warning(f"Selected too many features for plotting (> {MAX_FEATURES_FOR_PLOTTING}). Reduce features.")
        return

    df_class = df[features + [target]].dropna().reset_index(drop=True)
    if df_class.shape[0] < 15:
        st.warning("Not enough rows after removing NAs. Need ≥ 15 rows.")
        return

    numeric_features = [c for c in features if c in numeric_cols]
    if len(numeric_features) == 0:
        st.warning("Select at least one numeric feature for visuals.")
        return

    # Encode X + y
    X_encoded = pd.get_dummies(df_class[features], drop_first=True)
    y_cat = df_class[target].astype("category")
    y = y_cat.cat.codes.values
    class_names = list(y_cat.cat.categories)

    # ------------------ Class threshold ------------------
    if too_many_classes(y):
        st.warning(f"Dataset has too many classes (> {MAX_CLASSES}). Reduce classes for classification.")
        return

    # Safe train/test split
    X_train_enc, X_test_enc, y_train, y_test = safe_train_test_split(X_encoded, y)
    if X_train_enc is None:
        return

    # ------------------ Model Setup ------------------
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }

    choice = st.selectbox("Choose Model", list(models.keys()) + ["Compare All"])

    # ------------------ Evaluation Helper ------------------
    def evaluate(model):
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_test_enc)
        cm = confusion_matrix(y_test, preds)
        acc = accuracy_score(y_test, preds)
        return acc, cm, model

    # ------------------ Confusion Matrix ------------------
    def plot_cm(cm, labels):
        fig = ff.create_annotated_heatmap(
            cm, x=labels, y=labels, colorscale="Blues", showscale=True
        )
        fig.update_layout(width=550, height=550)
        st.plotly_chart(fig, use_container_width=False)

    # ------------------ Decision Tree Visual ------------------
    def decision_tree_visual():
        dt = DecisionTreeClassifier(random_state=42, max_depth=4)
        dt.fit(X_train_enc, y_train)

        tree_ = dt.tree_
        feature_names = X_encoded.columns

        if tree_.node_count > 70:
            st.warning("Tree too large to draw. Showing text summary instead.")
            st.text(export_text(dt, feature_names=list(feature_names)))
            return

        # Build coordinates
        x, y_coords, labels, edges = {}, {}, {}, []

        def walk(node, depth=0, xpos=0):
            y_coords[node] = -depth
            x[node] = xpos

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                fname = feature_names[tree_.feature[node]]
                labels[node] = f"{fname} ≤ {tree_.threshold[node]:.2f}"
            else:
                cls = class_names[tree_.value[node][0].argmax()]
                labels[node] = f"{cls}"

            left, right = tree_.children_left[node], tree_.children_right[node]
            if left != -1:
                edges.append((node, left))
                walk(left, depth + 1, xpos - 0.7 / (depth + 1))
            if right != -1:
                edges.append((node, right))
                walk(right, depth + 1, xpos + 0.7 / (depth + 1))

        walk(0)

        # Plotly graph
        edge_x, edge_y = [], []
        for p, c in edges:
            edge_x += [x[p], x[c], None]
            edge_y += [y_coords[p], y_coords[c], None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1)))
        fig.add_trace(go.Scatter(
            x=list(x.values()), y=list(y_coords.values()),
            mode="markers+text",
            text=list(labels.values()),
            textposition="top center",
            marker=dict(size=22, color="lightblue")
        ))
        fig.update_layout(width=650, height=750, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ KNN Visual ------------------
    def knn_visual_adaptive():
        X_num = df_class[numeric_features].astype(float).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_num)
        n = Xs.shape[1]

        knn_model = KNeighborsClassifier()
        knn_model.fit(Xs, y)

        # 2D CASE
        if n == 2:
            f1, f2 = numeric_features[:2]
            x_min, x_max = Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5
            y_min, y_max = Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                                 np.linspace(y_min, y_max, 250))
            Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig = go.Figure()
            fig.add_trace(go.Contour(
                z=Z, x=np.linspace(x_min, x_max, 250),
                y=np.linspace(y_min, y_max, 250),
                colorscale="Blues", opacity=0.40, showscale=False
            ))

            for i, cname in enumerate(class_names):
                mask = (y == i)
                fig.add_trace(go.Scatter(
                    x=Xs[mask, 0], y=Xs[mask, 1],
                    mode="markers", name=cname
                ))

            fig.update_layout(
                title="KNN — 2D Decision Boundary",
                width=1200, height=600,
                xaxis_title=f1, yaxis_title=f2
            )
            st.plotly_chart(fig)

        # 3D CASE
        elif n >= 3:
            if n > 3:
                pca = PCA(n_components=3)
                X3 = pca.fit_transform(Xs)
                labels_axes = ["PC1", "PC2", "PC3"]
            else:
                X3 = Xs[:, :3]
                labels_axes = numeric_features[:3]

            fig = go.Figure()
            for i, cname in enumerate(class_names):
                mask = (y == i)
                fig.add_trace(go.Scatter3d(
                    x=X3[mask, 0], y=X3[mask, 1], z=X3[mask, 2],
                    mode="markers", name=cname
                ))

            fig.update_layout(
                title="KNN — 3D Feature Scatter",
                width=1200, height=600,
                scene=dict(
                    xaxis_title=labels_axes[0],
                    yaxis_title=labels_axes[1],
                    zaxis_title=labels_axes[2]
                )
            )
            st.plotly_chart(fig)

    # ------------------ Single Model Mode ------------------
    if choice != "Compare All":
        acc, cm, trained = evaluate(models[choice])
        st.success(f"Accuracy: {acc*100:.2f}%")
        plot_cm(cm, class_names)

        if choice == "Decision Tree":
            decision_tree_visual()
            decision_tree_summary(trained, X_train_enc, y_train, X_test_enc, y_test, class_names)
        elif choice == "KNN":
            knn_visual_adaptive()
            knn_summary(trained, X_train_enc, y_train, X_test_enc, y_test, class_names)
        return

    # ------------------ Compare All ------------------
    st.subheader("Compare All Models")
    results = []
    for name, model in models.items():
        st.markdown(f"### {name}")
        acc, cm, trained = evaluate(model)
        results.append({"Model": name, "Accuracy": round(acc*100,2)})
        st.info(f"Accuracy: {acc*100:.2f}%")
        plot_cm(cm, class_names)

        if name == "Decision Tree":
            decision_tree_visual()
            decision_tree_summary(trained, X_train_enc, y_train, X_test_enc, y_test, class_names)
        elif name == "KNN":
            knn_visual_adaptive()
            knn_summary(trained, X_train_enc, y_train, X_test_enc, y_test, class_names)

    st.subheader("Summary Table")
    st.table(pd.DataFrame(results))
