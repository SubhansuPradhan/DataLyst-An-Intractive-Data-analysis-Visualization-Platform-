"""
Forecasting tab: uses Prophet if available, else shows message.
"""
import streamlit as st
import pandas as pd
import numpy as np

try:
    from prophet import Prophet  # if installed
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

def forecasting_tab(df: pd.DataFrame, numeric_cols):
    st.markdown("## Forecasting")

    # find datetime columns
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
    # also try to coerce object -> datetime (safe)
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = pd.to_datetime(df[c], errors="raise")
            if c not in dt_cols:
                dt_cols.append(c)
        except Exception:
            pass

    if not dt_cols:
        st.warning("No datetime column detected for forecasting.")
        return

    time_col = st.selectbox("Time column", dt_cols, key="time_col")
    if not numeric_cols:
        st.info("No numeric columns to forecast.")
        return
    target = st.selectbox("Target to forecast", numeric_cols, key="forecast_target")

    df_train = df[[time_col, target]].dropna().rename(columns={time_col: "ds", target: "y"})

    if HAS_PROPHET:
        st.info("Using Prophet for forecasting.")
        m = Prophet()
        m.fit(df_train)
        periods = st.number_input("Periods to forecast", min_value=1, max_value=1000, value=10)
        future = m.make_future_dataframe(periods=int(periods))
        forecast = m.predict(future)
        st.line_chart(forecast.set_index("ds")["yhat"].tail(periods))
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))
    elif HAS_SM:
        st.info("Prophet not available — using ARIMA (statsmodels).")
        series = df_train.set_index("ds")["y"].asfreq(pd.infer_freq(df_train["ds"]))
        # simple ARIMA(1,1,1)
        try:
            model = sm.tsa.ARIMA(series.dropna(), order=(1,1,1))
            res = model.fit()
            periods = st.number_input("Periods to forecast", min_value=1, max_value=1000, value=10)
            forecast = res.get_forecast(steps=int(periods))
            pred = forecast.predicted_mean
            st.line_chart(pred)
            st.dataframe(pred.to_frame(name="forecast").tail(int(periods)))
        except Exception as e:
            st.error(f"ARIMA failed: {e}")
    else:
        st.error("Neither Prophet nor statsmodels available. Install prophet or statsmodels for forecasting.")

    st.subheader("📘 Forecasting Summary")

    st.markdown("""
        This forecasting module predicts the future values of your selected numeric column
        based on the chosen time column. The forecast is displayed using a **simple line graph**
        showing only the predicted values for upcoming periods.

        ### 🔍 What the Graph Shows
        **Prophet Mode (if installed):**
        - The line chart displays **`yhat`**, the expected future value.
        - No actual vs predicted comparison is shown — only future predictions.
        - The forecast table includes:
            - `yhat` → main prediction  
            - `yhat_lower` → minimum expected value  
            - `yhat_upper` → maximum expected value  

        **ARIMA Mode (fallback):**
        - The line chart shows the **predicted mean values** for future steps.
        - Only future predictions are displayed — the graph does not show past values.
        - A simple forecast table is shown with the upcoming predicted values.

        ### 📈 How to Interpret the Forecast Line
        - An **upward slope** suggests increasing values ahead.
        - A **downward slope** indicates decline.
        - A **flat line** suggests stability or no detectable trend.

        ### 🧠 Model Behavior (Based on Your Code)
        - Prophet is used automatically if available — ideal for business-like time series.
        - ARIMA is used as a fallback — suitable for simpler or stable datasets.
        - No seasonal, trend, or component graphs are generated.
        - Only the **future forecast window** is visualized.

        ### ✔️ What This Helps You Understand
        - The expected trajectory of your selected metric.
        - The approximate range of uncertainty around predicted values.
        - Short-term movement and direction of your time-based data.

        This summary directly reflects the exact output of your forecasting graphs:  
        a **clean predicted line** with a **table of future values and confidence bounds**.
        """)