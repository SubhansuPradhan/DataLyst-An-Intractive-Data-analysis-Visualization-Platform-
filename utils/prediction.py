"""
Regression / prediction utilities used by Prediction tab.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def prediction_tab(df: pd.DataFrame, numeric_cols):
    if df is None:
        st.info("No data loaded.")
        return

    st.markdown("## Predictive Modeling")
    if not numeric_cols:
        st.info("Need numeric columns for regression.")
        return

    target = st.selectbox("Select target (Y)", numeric_cols, key="pred_target")
    features = st.multiselect("Select features (X)", [c for c in numeric_cols if c != target], key="pred_features")

    if not target or not features:
        st.info("Select target and at least one feature.")
        return

    model_choice = st.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest", "Compare All"], key="pred_model")
    df_model = df[features + [target]].dropna()
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    def evaluate(name, model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds)  # RMSE
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
        metrics = {"Model": name, "RMSE": round(rmse, 3), "MSE": round(mse, 3),
                   "R²": round(r2, 3), "MAE": round(mae, 3)}
        return result_df, metrics

    if model_choice != "Compare All":
        result_df, metrics = evaluate(model_choice, models[model_choice])
        st.markdown("### Model Evaluation Metrics")
        st.table(pd.DataFrame([metrics]))
        st.plotly_chart(px.line(result_df.reset_index(drop=True), y=["Actual", "Predicted"], markers=True),
                        use_container_width=True)
    else:
        summaries = []
        for name, model in models.items():
            _, metrics = evaluate(name, model)
            summaries.append(metrics)
        summary_df = pd.DataFrame(summaries)
        st.markdown("### Comparison of All Models")
        st.table(summary_df)

        # Identify the best model based on R² (higher is better)
        best_idx = summary_df['R²'].idxmax()
        best_model = summary_df.loc[best_idx, 'Model']
        best_r2 = summary_df.loc[best_idx, 'R²']
        st.markdown(f"**Best Model:** {best_model} (R² = {best_r2})")
        st.markdown(f"This model performs best because it explains the highest proportion of variance in the target variable. "
                    f"A higher R² indicates that predictions are closer to actual values compared to other models. "
                    f"Other metrics like RMSE and MAE can also be used for further evaluation of prediction errors.")
